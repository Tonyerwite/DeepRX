import argparse
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deeprx.matlab_bridge import (
    MatlabDeepRxBridge,
    OfficialBatch,
    PaperFigure6Config,
    paper_dataset_iteration,
    sample_paper_dataset_parameters,
)
from deeprx.model import DeepRx, DeepRxLoss, compute_ber


class Lamb(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), eps=1e-6, weight_decay=1e-4):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("Lamb does not support sparse gradients")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                adam_step = exp_avg.div(bias_correction1) / (exp_avg_sq.div(bias_correction2).sqrt().add(group["eps"]))
                if group["weight_decay"] != 0:
                    adam_step = adam_step.add(param, alpha=group["weight_decay"])

                weight_norm = torch.linalg.vector_norm(param)
                adam_norm = torch.linalg.vector_norm(adam_step)
                if weight_norm == 0 or adam_norm == 0 or not torch.isfinite(weight_norm) or not torch.isfinite(adam_norm):
                    trust_ratio = 1.0
                else:
                    trust_ratio = float(weight_norm / adam_norm)
                param.add_(adam_step, alpha=-group["lr"] * trust_ratio)
        return loss


def paper_learning_rate(step: int, *, total_steps: int, base_lr: float, warmup_steps: int, decay_start_fraction: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    decay_start = int(math.floor(total_steps * decay_start_fraction))
    if step < decay_start:
        return base_lr
    if total_steps <= decay_start:
        return 0.0
    progress = float(step - decay_start + 1) / float(total_steps - decay_start)
    return max(0.0, base_lr * (1.0 - progress))


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train PyTorch DeepRx using official MATLAB/5G Toolbox PUSCH batches.")
    parser.add_argument("--steps", type=int, default=30000, help="Paper training iterations.")
    parser.add_argument("--n-frames", type=int, default=8, help="MATLAB frames per step. With 10 TTIs/frame, 8 frames gives the paper's 80 TTI batch.")
    parser.add_argument("--optimizer", choices=("lamb", "adamw"), default="lamb")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=800)
    parser.add_argument("--decay-start-fraction", type=float, default=0.3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="checkpoints/deeprx_official_matlab.pt")
    parser.add_argument("--resume", default="")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
    return parser


def paper_training_frame_specs(config: PaperFigure6Config, *, step: int, n_frames: int, seed: int):
    start_index = step * n_frames
    specs = []
    for frame_offset in range(n_frames):
        index = start_index + frame_offset
        specs.append(
            (
                sample_paper_dataset_parameters(config, split="train", index=index, seed=seed),
                paper_dataset_iteration(config, split="train", index=index),
            )
        )
    return specs


def generate_paper_training_batch(
    bridge: MatlabDeepRxBridge,
    config: PaperFigure6Config,
    rng: random.Random,
    *,
    step: int,
    n_frames: int,
    seed: int,
) -> OfficialBatch:
    del rng
    batches = [
        bridge.generate_training_batch(parameters, iteration=iteration, n_frames=1)
        for parameters, iteration in paper_training_frame_specs(config, step=step, n_frames=n_frames, seed=seed)
    ]
    return _concat_official_batches(batches)


def main():
    args = build_arg_parser().parse_args()

    device = torch.device(args.device)
    rng = random.Random(args.seed)
    model = DeepRx(n_rx_antennas=2, max_bits_per_symbol=4).to(device)
    optimizer = _make_optimizer(args, model)
    criterion = DeepRxLoss()
    history = {"step": [], "loss": [], "ber": [], "lr": []}
    start_step = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = checkpoint.get("history", history)
        start_step = int(checkpoint.get("steps", 0))

    config = PaperFigure6Config()
    with MatlabDeepRxBridge() as bridge:
        for step in range(start_step, args.steps):
            lr = paper_learning_rate(
                step,
                total_steps=args.steps,
                base_lr=args.lr,
                warmup_steps=args.warmup_steps,
                decay_start_fraction=args.decay_start_fraction,
            )
            _set_optimizer_lr(optimizer, lr)
            frame_specs = paper_training_frame_specs(config, step=step, n_frames=args.n_frames, seed=args.seed)
            batch = generate_paper_training_batch(bridge, config, rng, step=step, n_frames=args.n_frames, seed=args.seed)
            inputs = batch.inputs.to(device)
            targets = batch.target_bits.to(device)
            data_mask = batch.data_mask.to(device)
            bit_mask = batch.bit_mask.to(device)

            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets, data_mask, bit_mask)
            loss.backward()
            optimizer.step()

            if step % args.log_every == 0:
                params = frame_specs[0][0]
                ber = compute_ber(logits.detach(), targets, data_mask, bit_mask)
                history["step"].append(step)
                history["loss"].append(float(loss.detach().cpu()))
                history["ber"].append(float(ber))
                history["lr"].append(float(lr))
                print(f"step={step:06d} loss={loss.item():.5f} ber={ber:.5f} lr={lr:.3e} snr={params.snr_db:+.2f} channel={params.channel_model}")

            if args.save_every > 0 and (step + 1) % args.save_every == 0:
                _save_checkpoint(args.output, model, optimizer, step + 1, args, history)

    _save_checkpoint(args.output, model, optimizer, args.steps, args, history)
    print(f"Saved {args.output}")


def _make_optimizer(args, model):
    if args.optimizer == "lamb":
        return Lamb(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def _set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def _concat_official_batches(batches):
    if not batches:
        raise ValueError("At least one batch is required")
    return OfficialBatch(
        inputs=torch.cat([batch.inputs for batch in batches], dim=0),
        target_bits=torch.cat([batch.target_bits for batch in batches], dim=0),
        data_mask=torch.cat([batch.data_mask for batch in batches], dim=0),
        bit_mask=batches[0].bit_mask,
    )


def _save_checkpoint(path, model, optimizer, step, args, history):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    state_dict_output = output.with_name(f"{output.stem}_state_dict.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "steps": step,
            "args": vars(args),
            "history": history,
        },
        output,
    )
    torch.save(model.state_dict(), state_dict_output)


if __name__ == "__main__":
    main()
