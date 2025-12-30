import copy
import argparse
import os
import torch
import torch.nn.functional as F
import torchaudio
import pytorch_lightning as pl
import transformers
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

# Local imports
from dataset.dcase25 import get_training_set, get_test_set
from helpers.init import worker_init_fn
from helpers.utils import mixstyle
from helpers import complexity
from models.net import get_model
from models.multi_device_model import MultiDeviceModelContainer


class PLModule(pl.LightningModule):
    """
    PyTorch Lightning Module for fine-tuning the Audio Mamba model for specific devices.
    """
    def __init__(self, config, base_model_state_dict=None):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        # ----- Preprocessing Pipeline -----
        self.mel = torch.nn.Sequential(
            torchaudio.transforms.Resample(
                orig_freq=config.orig_sample_rate,
                new_freq=config.sample_rate
            ),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=config.sample_rate,
                n_fft=config.n_fft,
                win_length=config.window_length,
                hop_length=config.hop_length,
                n_mels=config.n_mels,
                f_min=config.f_min,
                f_max=config.f_max
            )
        )
        self.mel_augment = torch.nn.Sequential(
            torchaudio.transforms.FrequencyMasking(config.freqm, iid_masks=True),
            torchaudio.transforms.TimeMasking(config.timem, iid_masks=True)
        )

        # ----- Metadata Definitions -----
        self.train_device_ids = ['a', 'b', 'c', 's1', 's2', 's3'] 
        self.device_ids = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']
        self.label_ids = ['airport', 'bus', 'metro', 'metro_station', 'park',
                          'public_square', 'shopping_mall', 'street_pedestrian',
                          'street_traffic', 'tram']
        self.device_groups = {
            'a': "real", 'b': "real", 'c': "real",
            's1': "seen", 's2': "seen", 's3': "seen",
            's4': "unseen", 's5': "unseen", 's6': "unseen"
        }

        # ----- Base Model Initialization (ALIGNED with train_base) -----
        base_model = get_model(
            n_classes=config.n_classes,
            n_mels=config.n_mels,
            target_length=config.target_length,
            embed_dim=config.embed_dim,
            depth=config.depth,
            patch_size=config.patch_size
        )

        if base_model_state_dict is not None:
            # We use strict=False to ignore the classification head if it differs
            base_model.load_state_dict(base_model_state_dict, strict=False)
            print("Successfully loaded pre-trained base model weights.")

        # Wrap in Multi-Device container
        self.multi_device_model = MultiDeviceModelContainer(
            base_model,
            self.train_device_ids
        )

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.validation_device = None

    def mel_forward(self, x):
        x = self.mel(x)
        if self.training:
            x = self.mel_augment(x)
        x = (x + 1e-5).log()
        return x

    def forward(self, x, devices):
        x = self.mel_forward(x)
        # Audio Mamba expects (B, 1, Freq, Time)
        if x.dim() == 3:
            x = x.unsqueeze(1) 
        return self.multi_device_model(x, devices)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, train_batch, batch_idx):
        x, _, labels, devices, _ = train_batch
        y_hat = self.forward(x, devices)
        loss = F.cross_entropy(y_hat, labels)
        
        self.log(f"train/loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, files, labels, devices, _ = val_batch
        # In multi-device FT, we expect batches to be sorted by device
        self.validation_device = devices[0] 
        y_hat = self.forward(x, devices)
        
        loss = F.cross_entropy(y_hat, labels)
        _, preds = torch.max(y_hat, dim=1)
        acc = (preds == labels).float().mean()

        results = {"loss": loss, "acc": acc, "count": torch.tensor(len(labels))}
        self.validation_step_outputs.append(results)
        return results

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
        avg_acc = torch.stack([x["acc"] for x in self.validation_step_outputs]).mean()
        avg_loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        
        if self.validation_device:
            self.log(f"val/acc.{self.validation_device}", avg_acc)
            self.log(f"val/loss.{self.validation_device}", avg_loss)
        
        self.validation_step_outputs.clear()

    def test_step(self, test_batch, batch_idx):
        # FIX: Removed manual .half() - handled by trainer precision arg
        x, files, labels, devices, _ = test_batch
        y_hat = self.forward(x, devices)
        
        _, preds = torch.max(y_hat, dim=1)
        results = {"preds": preds.cpu(), "labels": labels.cpu(), "devices": devices}
        self.test_step_outputs.append(results)

    def on_test_epoch_end(self):
        all_preds = torch.cat([x["preds"] for x in self.test_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.test_step_outputs])
        all_devices = [d for x in self.test_step_outputs for d in x["devices"]]
        
        device_accs = []
        for d in self.device_ids:
            mask = torch.tensor([dev == d for dev in all_devices])
            if mask.any():
                acc = (all_preds[mask] == all_labels[mask]).float().mean()
                self.log(f"test/acc.{d}", acc)
                device_accs.append(acc)
        
        if device_accs:
            self.log("test/macro_avg_acc", torch.stack(device_accs).mean())
        self.test_step_outputs.clear()


def train(config):
    # 1. Resolve Best Checkpoint Path
    base_model_state_dict = None
    if config.ckpt_id:
        ckpt_dir = os.path.join(config.project_name, config.ckpt_id, "checkpoints")
        if os.path.exists(ckpt_dir):
            ckpts = [f for f in os.listdir(ckpt_dir) if "best" in f and f.endswith(".ckpt")]
            ckpt_path = os.path.join(ckpt_dir, ckpts[0] if ckpts else "last.ckpt")
            print(f"Loading weights from: {ckpt_path}")
            
            full_ckpt = torch.load(ckpt_path, map_location="cpu")
            # Filter state dict for the base model part
            base_model_state_dict = {
                k.replace("model.", ""): v for k, v in full_ckpt["state_dict"].items() 
                if k.startswith("model.") and "classifier" not in k
            }

    # 2. Setup Logger
    wandb_logger = WandbLogger(
        project=config.project_name,
        name=config.experiment_name,
        config=config
    )

    pl_module = PLModule(config, base_model_state_dict=base_model_state_dict)

    # 3. Fine-tuning Loop per Device
    for device_id in pl_module.train_device_ids:
        print(f"\n>>> Fine-tuning for Device: {device_id.upper()}")
        
        train_dl = DataLoader(get_training_set(config.subset, device=device_id, roll=config.orig_sample_rate * config.roll_sec), 
                              batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)
        val_dl = DataLoader(get_test_set(device=device_id), 
                            batch_size=config.batch_size, num_workers=config.num_workers)

        trainer = pl.Trainer(
            max_epochs=config.n_epochs,
            logger=wandb_logger,
            accelerator="gpu",
            devices=1,
            precision=config.precision,
            enable_checkpointing=True
        )
        trainer.fit(pl_module, train_dl, val_dl)

    # 4. Final Evaluation on All Devices
    test_dl = DataLoader(get_test_set(device=None), batch_size=config.batch_size, num_workers=config.num_workers)
    trainer.test(pl_module, dataloaders=test_dl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Paths and Experiment Setup
    parser.add_argument("--ckpt_id", type=str, default=None, help="W&B ID or folder name for checkpoint")
    parser.add_argument("--project_name", type=str, default="DCASE25_Task1")
    parser.add_argument("--experiment_name", type=str, default="AUM_MultiDevice_FT")
    
    # Model Architecture (MUST MATCH train_base)
    parser.add_argument("--embed_dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--patch_size", type=str, default="16,16")
    parser.add_argument("--target_length", type=int, default=1024)
    parser.add_argument("--n_classes", type=int, default=10)

    # Fine-tuning Hyperparameters
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--n_epochs", type=int, default=50) 
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    
    # Preprocessing (MUST MATCH train_base)
    parser.add_argument("--orig_sample_rate", type=int, default=44100)
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--n_fft", type=int, default=4096)
    parser.add_argument("--window_length", type=int, default=3072)
    parser.add_argument("--hop_length", type=int, default=500)
    parser.add_argument("--n_mels", type=int, default=256)
    parser.add_argument("--f_min", type=int, default=0)
    parser.add_argument("--f_max", type=int, default=None)
    parser.add_argument("--freqm", type=int, default=48)
    parser.add_argument("--timem", type=int, default=0)
    parser.add_argument("--mixstyle_p", type=float, default=0.0)
    parser.add_argument("--roll_sec", type=float, default=0.1)

    # System
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--subset", type=int, default=25)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=10)

    args = parser.parse_args()
    train(args)