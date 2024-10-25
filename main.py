import torch
import torch.optim as optim

import sys
import os

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet3d import UNet3D
from config.config import Config
from data.dataloader import get_patch_dataloader
from train.trainer import Trainer

def main():
    # Load configuration
    config = Config()
    config.epochs = 10
    config.batch_size = 4
    config.learning_rate = 1e-4

    # Initialize model, optimizer, and DataLoaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Loaders
    train_loader = get_patch_dataloader(config=config, train=True)
    val_loader = get_patch_dataloader(config=config, train=False)

    # Trainer setup
    trainer = Trainer(model, optimizer, train_loader, val_loader, config)

    # Start training
    print("[Debug] Starting Training")
    trainer.train()

if __name__ == '__main__':
    main()
