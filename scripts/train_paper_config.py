import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deeprx.data import DeepRxDataset, collate_samples
from deeprx.model import DeepRx, DeepRxLoss, compute_ber


def main():
    parser = argparse.ArgumentParser(description="Train DeepRx with paper-size tensor dimensions.")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-bits", type=int, default=4, help="4 matches the MathWorks 16QAM example; 8 enables 256QAM-style masking.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="checkpoints/deeprx_paper_config.pt")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = DeepRxDataset(
        n_samples=max(args.steps * args.batch_size, args.batch_size),
        max_bits_per_symbol=args.max_bits,
        device=str(device),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_samples)
    model = DeepRx(max_bits_per_symbol=args.max_bits).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = DeepRxLoss()
    history = {"step": [], "loss": [], "ber": []}

    step = 0
    for batch in loader:
        if step >= args.steps:
            break
        inputs = batch["inputs"].to(device)
        targets = batch["target_bits"].to(device)
        data_mask = batch["data_mask"].to(device)
        bit_mask = batch["bit_mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets, data_mask, bit_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.log_every == 0:
            ber = compute_ber(logits.detach(), targets, data_mask, bit_mask)
            history["step"].append(step)
            history["loss"].append(float(loss.detach().cpu()))
            history["ber"].append(float(ber))
            print(f"step={step:06d} loss={loss.item():.4f} ber={ber:.4f}")
        step += 1
        if args.save_every > 0 and step % args.save_every == 0:
            _save_checkpoint(args.output, model, optimizer, step, args, history)

    _save_checkpoint(args.output, model, optimizer, step, args, history)
    print(f"Saved {args.output}")


def _save_checkpoint(path, model, optimizer, step, args, history):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    main()
