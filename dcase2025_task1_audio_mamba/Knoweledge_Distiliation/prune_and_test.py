import torch
import torch.nn.utils.prune as prune
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import argparse

# Local imports
from train_distillation import DistillationModule
from dataset.dcase25 import get_test_set

def count_non_zero_params(model):
    """Calculates exactly how many parameters are not zero."""
    non_zeros = 0
    total = 0
    for param in model.parameters():
        if param.requires_grad:
            non_zeros += torch.count_nonzero(param).item()
            total += param.numel()
    return non_zeros, total

def prune_model(model, prune_amount):
    """Applies global unstructured pruning to all Linear and Conv layers."""
    parameters_to_prune = []
    
    # Identify all dense and convolutional layers in the student model
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv1d) or isinstance(module, torch.nn.Conv2d):
            parameters_to_prune.append((module, 'weight'))

    # Apply global magnitude pruning (removes the smallest X% of weights across the whole model)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=prune_amount,
    )

    # Make the pruning permanent
    for module, name in parameters_to_prune:
        prune.remove(module, name)
        
    return model

def main(args):
    print(f"\n--- 1. Loading Trained Model ---")
    print(f"Checkpoint: {args.ckpt_path}")
    
    # Load the Lightning Module from your 44% run
    pl_module = DistillationModule.load_from_checkpoint(args.ckpt_path)
    student = pl_module.student
    student.eval()

    # Pre-Pruning Stats
    nz_pre, total_pre = count_non_zero_params(student)
    print(f"Pre-Surgery: {total_pre:,} parameters ({(total_pre)/1024:.2f} KB in INT8)")

    print(f"\n--- 2. Performing Surgery (Pruning {args.prune_amount * 100}%) ---")
    # Apply Pruning
    student = prune_model(student, args.prune_amount)
    
    # Post-Pruning Stats
    nz_post, total_post = count_non_zero_params(student)
    print(f"Post-Surgery Non-Zero Params: {nz_post:,}")
    print(f"DCASE INT8 Submission Size: {nz_post / 1024:.2f} KB / 128.00 KB Limit")

    if nz_post <= 128000:
        print("✅ MODEL IS LEGAL FOR DCASE!")
    else:
        print("❌ MODEL IS STILL OVER THE LIMIT.")

    print(f"\n--- 3. Testing Brain Damage (Accuracy Drop) ---")
    # Setup Trainer for quick testing
    trainer = pl.Trainer(accelerator="gpu", devices=1, logger=False)
    test_dl = DataLoader(get_test_set(), num_workers=4, batch_size=64)
    
    # Run the official test loop
    trainer.validate(pl_module, dataloaders=test_dl)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to your best-student checkpoint")
    parser.add_argument("--prune_amount", type=float, default=0.54, help="Percentage of weights to zero out")
    args = parser.parse_args()
    main(args)