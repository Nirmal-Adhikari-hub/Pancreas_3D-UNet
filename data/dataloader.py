import os
import torch
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from monai.transforms import (
    RandFlip, RandRotate90, RandScaleIntensity, RandZoom, Compose, RandGaussianNoise
    # RandFlip, RandRotate90, RandScaleIntensity, RandZoom, Compose, RandGaussianNoise, RandElasticDeformation
)
import nibabel as nib  # To load the NIfTI files
import sys
import logging

# Logger class to log stdout and stderr to both terminal and file
class TeeLogger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = open(log_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

# Set up the log file and redirect stdout/stderr
log_filename = "logs.log"
sys.stdout = TeeLogger(log_filename)
sys.stderr = TeeLogger(log_filename)

# Add the parent directory to the Python path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.distributed import get_patch_slices, pad_if_needed


class PancreasDataset(Dataset):
    def __init__(self, config, train=True):
        """
        Pancreas Dataset for loading and augmenting 3D CT scans dynamically.
        :param config: Configuration object containing paths, batch_size, etc.
        :param train: Boolean to indicate training or validation mode.
        """
        self.config = config
        self.train = train

        # Load dataset info (dataset.json)
        with open(config.dataset_json, 'r') as f:
            dataset_info = json.load(f)

        self.data_info = dataset_info['training'] if train else dataset_info['test']

        # Augmentation configuration based on number of augmented samples
        self.augmented_samples = config.augmented_samples  # New parameter to control augmented samples

        if self.train:
            # Augmentations based on best practices in medical imaging:
            self.transforms = Compose([
                RandFlip(spatial_axis=[1], prob=0.5),  # Flip along x-axis
                RandFlip(spatial_axis=[2], prob=0.5),  # Flip along y-axis
                RandFlip(spatial_axis=[0], prob=0.5),  # Flip along z-axis
                # RandRotate90(prob=0.5, max_k=3, spatial_axes=(1, 2)),  # Random 90-degree rotation on XY plane
                # RandScaleIntensity(factors=0.1, prob=0.5),  # Random intensity scaling
                # RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),  # Random zooming
                # RandGaussianNoise(prob=0.2),  # Adding Gaussian noise
                # RandElasticDeformation(prob=0.3, sigma_range=(5, 10))  # Elastic deformation for tissue-like distortion
            ]) if self.augmented_samples > 1 else None
        else:
            self.transforms = None  # No augmentation for validation or test data

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        # Get file paths for the image and the corresponding label
        img_path = os.path.join(self.config.dataset_path, self.data_info[idx]['image'])
        label_path = os.path.join(self.config.dataset_path, self.data_info[idx]['label'])

        # Load the NIfTI files (3D medical imaging format)
        img_nifti = nib.load(img_path)
        label_nifti = nib.load(label_path)

        # Convert to numpy arrays and ensure correct axes order
        image = np.transpose(np.array(img_nifti.get_fdata(), dtype=np.float32), (2, 0, 1))  # Shape: (D, H, W)
        label = np.transpose(np.array(label_nifti.get_fdata(), dtype=np.uint8), (2, 0, 1))  # Shape: (D, H, W)

        # Apply patch extraction based on the configuration settings
        depth, height, width = self.config.input_size
        patches, labels = self.get_patches(image, label, depth, height, width)

        print(f"Number of patches: {len(patches)}, \n Number of Labels: {len(labels)}")

        augmented_patches, augmented_labels = [], []

        if self.train:
            for patch, label_patch in zip(patches, labels):
                # Always add the original scan first
                augmented_patches.append(torch.tensor(patch))  # Original scan
                augmented_labels.append(torch.tensor(label_patch))  # Convert original label to tensor
                
                # Generate the required number of augmented samples
                for _ in range(self.config.augmented_samples - 1):
                    # print(f"Original patch shape: {patch.shape}")
                    if self.transforms:                        
                        # Apply augmentation (operates on NumPy arrays)
                        aug_patch = self.transforms(patch)

                        # Assert the shape after augmentation
                        assert aug_patch.shape == (32, 512, 512), f"Augmented patch shape mismatch: expected (32, 512, 512), got {aug_patch.shape}"

                        
                        # Convert augmented patch back to tensor
                        aug_patch_tensor = torch.from_numpy(aug_patch)
                        # print(f"Augmented patch shape: {aug_patch_tensor.shape}")
                        
                        # Append the augmented patch and the corresponding label tensor
                        augmented_patches.append(aug_patch_tensor)
                        augmented_labels.append(torch.tensor(label_patch))  # Convert label to tensor (consistent with patch)

            # Stack the augmented patches and labels
            print(f"After Augmentation: \n Patches - {torch.stack(augmented_patches).shape}, Labels -> {torch.stack(augmented_labels).shape}")

            return torch.stack(augmented_patches), torch.stack(augmented_labels)
        else:
            return torch.stack(patches), torch.stack(labels)


    def get_patches(self, image, label, depth, height, width):
        """
        Extract overlapping patches from 3D volume and corresponding labels.
        :param image: 3D numpy array (CT scan)
        :param label: 3D numpy array (Segmentation mask)
        :return: Patches from image and corresponding label patches
        """
        image = pad_if_needed(image, depth)
        label = pad_if_needed(label, depth)
        patch_slices = get_patch_slices(image.shape, depth, self.config.patch_overlap)
        patches, labels = [], []

        for sl in patch_slices:
            patches.append(image[sl].copy())
            labels.append(label[sl].copy())

        return patches, labels


# Dataloader function to return DataLoader objects for training and validation
def get_dataloaders(config):
    train_dataset = PancreasDataset(config, train=True)
    val_dataset = PancreasDataset(config, train=False)

    # Get the distributed rank and world size (for distributed training)
    rank = config.get_local_rank()
    world_size = config.get_world_size()
    if torch.distributed.get_rank() == 0:  # Ensure only the main process logs
        print(f"World Size: {world_size} \n Local Rank: {rank}")

    # For distributed data loading
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    # Train and validation dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, sampler=val_sampler, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.config import Config
    from models.unet3d import UNet3D
    from data.dataloader import get_dataloaders

    config = Config()

    config.init_distributed()

    # Set the correct local rank for each process
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    print(f"Using {config.num_gpus} GPUs on {config.device}")

    print(f"DIST.IS_AVAILABLE(): {dist.is_available()} DIST.IS_INITIALIZED(): {dist.is_initialized()}")

    # Initialize dataloaders
    train_loader, val_loader = get_dataloaders(config)

    # Initialize the model, DDP wraps it around
    model = UNet3D(config.in_channels, config.num_classes).to(config.device)
    if config.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])


    # Testing with one batch from the train_loader
    for batch_data, batch_labels in train_loader:
        print(f"Batch data shape: {batch_data.shape}")  # Expected shape: (Batch, Patches, C, D, H, W)
        print(f"Batch labels shape: {batch_labels.shape}")  # Expected shape: (Batch, Patches, D, H, W)
        break  # Just run one batch to check
