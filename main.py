import torch
import torch.optim as optim
import torch.distributed as dist
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

    # Initialize distributed process group
    dist.init_process_group(backend="nccl")  # 'nccl' backend is optimal for multi-GPU training

    # Set device based on local rank
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Initialize model and move to device
    model = UNet3D(config).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Dataloader setup for distributed training
    train_loader = get_patch_dataloader(config=config, train=True, distributed=True)
    val_loader = get_patch_dataloader(config=config, train=False, distributed=True)

    # Trainer setup
    trainer = Trainer(model, optimizer, train_loader, val_loader, config)

    # Start training
    print("[Debug] Starting Training with Distributed Data Parallel")
    trainer.train()

    # Cleanup
    dist.destroy_process_group()

if __name__ == '__main__':
    main()