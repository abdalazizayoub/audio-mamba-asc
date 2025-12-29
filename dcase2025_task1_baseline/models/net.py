import torch
import torch.nn as nn
import sys
import os
import warnings

# Adjust path for local imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import AUM.src.models.mamba_models as mamba_models  # Import Audio Mamba models

class AudioMambaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Audio Mamba configuration
        self.spectrogram_size = (config['n_mels'], config['target_length']) 
        
        # Parse patch_size string "16,16" into a tuple (16, 16)
        patch_values = [int(v.strip()) for v in config['patch_size'].split(',')]
        self.patch_size = tuple(patch_values)
        self.strides = self.patch_size # Assuming strides equal patch size
        
        self.embed_dim = config['embed_dim']
        self.depth = config['depth']
        
        bimamba_type = 'v2' 
        
        # Initialize Audio Mamba model
        self.aum_model = mamba_models.AudioMamba(
            spectrogram_size=self.spectrogram_size,
            patch_size=self.patch_size,
            strides=self.strides,
            embed_dim=self.embed_dim,
            num_classes=config['n_classes'],
            imagenet_pretrain=False,
            imagenet_pretrain_path=None,
            aum_pretrain=False,  
            aum_pretrain_path=None, 
            bimamba_type=bimamba_type,
            if_bidirectional=True,
         
            
            # --- CRITICAL FIX ---
            # Disable absolute positional embeddings to bypass the problematic initialization
            if_abs_pos_embed=False,
            # Ensure RoPE is also disabled for simplest initialization
            # --------------------
        )
        
        # Remove the final classification head
        if hasattr(self.aum_model, 'head'):
            self.aum_model.head = nn.Identity()
        
        # Add a new classification head for DCASE25
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, config['n_classes'])
        )
        
        # Initialize classifier weights
        self._init_weights(self.classifier)
    
    def _init_weights(self, module):
        """Initialize weights for linear layers"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass for Audio Mamba
        x: input spectrogram of shape (batch, 1, n_mels, target_length)
        """
        # Pass through Audio Mamba backbone
        features = self.aum_model.forward_features(x)
        
        # Classification
        logits = self.classifier(features)
        return logits


def get_model(n_classes, n_mels, target_length, embed_dim, depth, patch_size):
    """
    Get Audio Mamba model for DCASE25 Task 1, using parameters passed from train_base.py
    """
    config = {
        'n_classes': n_classes,
        'n_mels': n_mels,
        'target_length': target_length,
        'embed_dim': embed_dim,
        'depth': depth,
        'patch_size': patch_size,
    }
    return AudioMambaModel(config)