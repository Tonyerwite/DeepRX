import torch

from deeprx.qam import QAMModem, max_log_llrs


class LMMSEReceiver:
    """Pilot LS estimation, linear interpolation, SIMO LMMSE equalization, max-log demapping."""

    def __init__(self, modulation: str = "16QAM", device: str = "cpu", max_bits_per_symbol: int = 8):
        self.modem = QAMModem(modulation, device=device)
        self.device = torch.device(device)
        self.max_bits_per_symbol = max_bits_per_symbol

    def estimate_channel_ls(self, rx_grid: torch.Tensor, tx_pilots: torch.Tensor, pilot_mask: torch.Tensor) -> torch.Tensor:
        n_rx = rx_grid.shape[1]
        return rx_grid * torch.conj(tx_pilots.expand(-1, n_rx, -1, -1)) * pilot_mask.to(rx_grid.device)

    def interpolate_channel(self, h_raw: torch.Tensor, pilot_mask: torch.Tensor) -> torch.Tensor:
        batch, n_rx, n_symbols, n_subcarriers = h_raw.shape
        mask = pilot_mask.to(h_raw.device)[0, 0].bool()
        out = torch.zeros_like(h_raw)
        freq_query = torch.arange(n_subcarriers, dtype=torch.float32, device=h_raw.device)

        pilot_symbols = [idx for idx in range(n_symbols) if bool(mask[idx].any())]
        for b in range(batch):
            for rx in range(n_rx):
                per_symbol = torch.zeros(n_symbols, n_subcarriers, dtype=torch.cfloat, device=h_raw.device)
                for symbol_idx in pilot_symbols:
                    pos = mask[symbol_idx].nonzero(as_tuple=True)[0]
                    vals = h_raw[b, rx, symbol_idx, pos]
                    per_symbol[symbol_idx] = _interp_complex(pos.float(), vals, freq_query)

                if len(pilot_symbols) == 1:
                    out[b, rx] = per_symbol[pilot_symbols[0]].unsqueeze(0).expand(n_symbols, -1)
                else:
                    time_known = torch.tensor(pilot_symbols, dtype=torch.float32, device=h_raw.device)
                    time_query = torch.arange(n_symbols, dtype=torch.float32, device=h_raw.device)
                    for subcarrier in range(n_subcarriers):
                        vals = per_symbol[pilot_symbols, subcarrier]
                        out[b, rx, :, subcarrier] = _interp_complex(time_known, vals, time_query)
        return out

    def estimate_noise_power(self, h_raw: torch.Tensor, pilot_mask: torch.Tensor) -> torch.Tensor:
        mask = pilot_mask.to(h_raw.device)[0, 0].bool()
        diffs = []
        for symbol_idx in range(mask.shape[0]):
            pos = mask[symbol_idx].nonzero(as_tuple=True)[0]
            if len(pos) > 1:
                vals = h_raw[:, :, symbol_idx, pos]
                diffs.append(vals[:, :, 1:] - vals[:, :, :-1])
        if not diffs:
            return torch.full((h_raw.shape[0],), 1e-3, device=h_raw.device)
        all_diffs = torch.cat([d.reshape(h_raw.shape[0], -1) for d in diffs], dim=1)
        sigma2 = (all_diffs.abs() ** 2).mean(dim=1) / 2.0
        return sigma2.clamp_min(1e-5)

    def equalize(self, rx_grid: torch.Tensor, h_est: torch.Tensor, noise_power: torch.Tensor):
        numerator = (torch.conj(h_est) * rx_grid).sum(dim=1)
        channel_power = (h_est.abs() ** 2).sum(dim=1)
        sigma2 = noise_power.view(-1, 1, 1)
        eq = numerator / (channel_power + sigma2).clamp_min(1e-8)
        eq_snr = channel_power / sigma2.clamp_min(1e-8)
        return eq, eq_snr

    def process(self, rx_grid: torch.Tensor, tx_pilots: torch.Tensor, pilot_mask: torch.Tensor, known_channel: torch.Tensor = None) -> torch.Tensor:
        h_raw = self.estimate_channel_ls(rx_grid, tx_pilots, pilot_mask)
        h_est = known_channel if known_channel is not None else self.interpolate_channel(h_raw, pilot_mask)
        noise_power = self.estimate_noise_power(h_raw, pilot_mask)
        eq, eq_snr = self.equalize(rx_grid, h_est, noise_power)
        return max_log_llrs(eq, eq_snr, self.modem, b_max=self.max_bits_per_symbol)


def _interp_complex(x_known: torch.Tensor, y_known: torch.Tensor, x_query: torch.Tensor) -> torch.Tensor:
    if len(x_known) == 1:
        return y_known[0].expand(len(x_query))
    real = _interp_real(x_known, y_known.real, x_query)
    imag = _interp_real(x_known, y_known.imag, x_query)
    return torch.complex(real, imag)


def _interp_real(x_known: torch.Tensor, y_known: torch.Tensor, x_query: torch.Tensor) -> torch.Tensor:
    idx = torch.searchsorted(x_known, x_query, right=True) - 1
    idx = idx.clamp(0, len(x_known) - 2)
    x0 = x_known[idx]
    x1 = x_known[idx + 1]
    y0 = y_known[idx]
    y1 = y_known[idx + 1]
    weight = (x_query - x0) / (x1 - x0).clamp_min(1e-8)
    return y0 + weight * (y1 - y0)
