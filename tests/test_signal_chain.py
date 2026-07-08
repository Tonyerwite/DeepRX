import torch

from deeprx.model import compute_ber, create_bit_mask, create_pilot_mask, generate_qpsk_pilots
from deeprx.ofdm import OFDMReceiver, OFDMTransmitter, add_awgn
from deeprx.qam import QAMModem
from deeprx.receiver import LMMSEReceiver


def test_qam_modem_round_trips_nearest_constellation_points():
    for modulation in ["QPSK", "16QAM", "64QAM", "256QAM"]:
        modem = QAMModem(modulation)
        bits = torch.randint(0, 2, (512, modem.bits_per_symbol)).float()
        symbols = modem.modulate(bits)
        recovered = modem.hard_demodulate(symbols)

        assert torch.allclose((symbols.abs() ** 2).mean(), torch.tensor(1.0), atol=0.15)
        assert torch.equal(bits, recovered)


def test_ofdm_frontend_round_trips_without_channel_or_noise():
    tx = OFDMTransmitter(n_subcarriers=72, n_fft=128, cp_length=16, n_symbols=14)
    rx = OFDMReceiver(n_subcarriers=72, n_fft=128, cp_length=16, n_symbols=14)
    grid = torch.randn(3, 1, 14, 72, dtype=torch.cfloat)

    waveform = tx.modulate(grid)
    recovered = rx.demodulate(waveform, n_rx=1)

    assert torch.allclose(recovered, grid, atol=1e-5)


def test_lmmse_receiver_has_low_ber_on_identity_channel_at_high_snr():
    batch, symbols, subcarriers = 2, 14, 72
    tx = OFDMTransmitter(n_subcarriers=subcarriers, n_fft=128, cp_length=16, n_symbols=symbols)
    rx = OFDMReceiver(n_subcarriers=subcarriers, n_fft=128, cp_length=16, n_symbols=symbols)
    modem = QAMModem("16QAM")
    pilot_mask = create_pilot_mask(symbols, subcarriers, "2_pilots_left")
    data_mask = 1.0 - pilot_mask
    bit_mask = create_bit_mask("16QAM")

    n_data = int(data_mask.sum().item())
    bits = torch.randint(0, 2, (batch, n_data, modem.bits_per_symbol)).float()
    symbols_data = modem.modulate(bits.reshape(-1, modem.bits_per_symbol)).reshape(batch, n_data)
    pilots = generate_qpsk_pilots(batch, symbols, subcarriers, pilot_mask)
    grid, target_bits = tx.build_resource_grid(symbols_data, pilots, pilot_mask, bits)
    waveform = tx.modulate(grid)
    noisy = add_awgn(waveform, 35.0)
    rx_grid = rx.demodulate(noisy.unsqueeze(1).expand(-1, 2, -1), n_rx=2)

    llrs = LMMSEReceiver("16QAM").process(rx_grid, pilots, pilot_mask)
    ber = compute_ber(llrs, target_bits, data_mask, bit_mask)

    assert ber < 0.02


def test_lmmse_receiver_can_emit_four_bit_16qam_llrs():
    batch, symbols, subcarriers = 1, 14, 72
    tx = OFDMTransmitter(n_subcarriers=subcarriers, n_fft=128, cp_length=16, n_symbols=symbols)
    rx = OFDMReceiver(n_subcarriers=subcarriers, n_fft=128, cp_length=16, n_symbols=symbols)
    modem = QAMModem("16QAM")
    pilot_mask = create_pilot_mask(symbols, subcarriers, "2_pilots_left")
    data_mask = 1.0 - pilot_mask
    n_data = int(data_mask.sum().item())
    bits = torch.randint(0, 2, (batch, n_data, modem.bits_per_symbol)).float()
    data_symbols = modem.modulate(bits.reshape(-1, modem.bits_per_symbol)).reshape(batch, n_data)
    pilots = generate_qpsk_pilots(batch, symbols, subcarriers, pilot_mask)
    grid, _ = tx.build_resource_grid(data_symbols, pilots, pilot_mask, bits, max_bits=4)
    rx_grid = rx.demodulate(tx.modulate(grid).unsqueeze(1).repeat(1, 2, 1), n_rx=2)

    llrs = LMMSEReceiver("16QAM", max_bits_per_symbol=4).process(rx_grid, pilots, pilot_mask)

    assert llrs.shape == (batch, 4, symbols, subcarriers)
