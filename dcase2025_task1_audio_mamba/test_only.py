import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import argparse
import os

# Local imports from your project
from dataset.dcase25 import get_test_set
from helpers.init import worker_init_fn
from train_base import PLModule  # Import the module structure from your training file

def test_best_model(args):
    # 1. Load the Test Dataloader
    test_ds = get_test_set(device=None)
    test_dl = DataLoader(
        dataset=test_ds,
        worker_init_fn=worker_init_fn,
        num_workers=args.num_workers,
        batch_size=args.batch_size
    )

    # 2. Reconstruct the PLModule with the same config
    # This ensures the model architecture (AuM-Small/Base) matches the checkpoint
    model = PLModule(args)

    # 3. Initialize the Trainer
    # We use precision=32 to match your successful training config
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=args.precision
    )

    # 4. Run the test using the specific checkpoint path
    # Replace 'args.ckpt_path' with the actual path to your .ckpt file
    print(f"--- Running Test on Checkpoint: {args.ckpt_path} ---")
    trainer.test(model, dataloaders=test_dl, ckpt_path=args.ckpt_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCASE 25 Test-Only Script')
    
    # CRITICAL: Path to your best checkpoint
    # Example: "DCASE25_Task1/AUM_Small_Scratch/checkpoints/best-aum-epoch=144-val/macro_avg_acc=0.50.ckpt"
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the best .ckpt file")

    # Metadata/Arch params (Must match the training run exactly)
    parser.add_argument("", type=int, default=384)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--patch_size", type=str, default="16,16")
    parser.add_argument("--n_mels", type=int, default=256)
    parser.add_argument("--target_length", type=int, default=1024)
    parser.add_argument("--n_classes", type=int, default=10)
    
    # System params
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--precision", type=str, default="32")
    
    # Preprocessing params (Must match training)
    parser.add_argument("--orig_sample_rate", type=int, default=44100)
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--window_length", type=int, default=3072)
    parser.add_argument("--hop_length", type=int, default=500)
    parser.add_argument("--n_fft", type=int, default=4096)
    parser.add_argument("--freqm", type=int, default=48)
    parser.add_argument("--timem", type=int, default=64)
    parser.add_argument("--f_min", type=int, default=0)
    parser.add_argument("--f_max", type=int, default=None)

    args = parser.parse_args()
    test_best_model(args)