import os
import json
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import numpy as np
import nibabel as nib
import torchio as tio
from torchio import SubjectsLoader

def load_npy_file(file_path):
    return np.load(file_path)


def load_nifti_file(file_path):
    nifti_img = nib.load(file_path)
    image = np.transpose(nifti_img.get_fdata(), (2, 0, 1)) # Convert to (z, x, y)
    return image, nifti_img.affine


# Dataset class for Train/Val patches
class PancreasPatchDataset(Dataset):
    def __init__(self, config, transform=None, train=True):
        """
        Custom Dataset for pancreas segmentation patches.
        :param dataset_json: Path to the JSON file that contains the list of .npy files (image/label paths).
        :param base_dir: Base directory to which the relative paths are appended.
        :param transform: Optional augmentations to apply.
        :param augment: Whether to apply augmentations during training.
        """
        self.config = config
        self.dataset_json = config.train_dataset_json if train else config.val_dataset_json
        self.base_dir = config.preprocessed_dir or os.path.dirname(self.dataset_json)
        with open(self.dataset_json, 'r') as f:
            self.data_info = json.load(f)  # Load .npy paths and metadata

        self.phase = "train" if train else "val"
        
        # Extract list of image/label pairs from JSON (for training and validation)
        self.data_list = self.data_info.get(self.phase, [])
        
        self.transform = transform
        self.augmented_samples = config.augmented_samples

    def __len__(self):
        return len(self.data_list) * self.augmented_samples
    
    def __getitem__(self, index):
        """
        Load a patch and its corresponding segmentation mask.
        The first sample is the original, and subsequent ones are augmented, if config.augmented_samples > 1
        """
        # Adjust the index based on the augmented samples
        original_index = index // self.augmented_samples
        augmentation_index = index % self.augmented_samples

        item_info = self.data_list[original_index]

        # Construct the full path by combining base_dir and relative paths
        image_path = os.path.join(self.base_dir, item_info["image"])
        label_path = os.path.join(self.base_dir, item_info["label"])

        # Load the image and label from .npy files
        image = load_npy_file(image_path)
        label = load_npy_file(label_path)

        # Convert to torch tensor
        image = torch.from_numpy(image).float().unsqueeze(0) # Add the channel dimension (1, 32, 512, 512)
        label = torch.from_numpy(label).long().unsqueeze(0) # Assuming labels are stores as integers

        # The datatype expected by the SubjectsLoader is tio type
        subject = tio.Subject(
                image=tio.ScalarImage(tensor=image), 
                label=tio.LabelMap(tensor=label)
            )
        
        if augmentation_index != 0 and self.transform:
            subject = self.transform(subject)

        # Apply augmentations if it's an augmented sample and augmentations are defined
        # if augmentation_index != 0 and self.transform:
        #     subject = tio.Subject(
        #         image=tio.ScalarImage(tensor=image), 
        #         label=tio.LabelMap(tensor=label)
        #     )
        #     transformed = self.transform(subject)
        #     image, label = transformed.image.tensor, transformed.label.tensor

        # image = image.float()
        # label = label.long()

        # return image, label
        return subject


# Dataset class for Test scans (full NIfTI volumes)
class PancreasTestDataset(Dataset):
    def __init__(self, config, transform=None):
        """
        Custom Dataser for full CT scans (test set).
        :param dataset_json: Path to JSON file containing paths to .nii.gz files
        :para base_dir: Base directory for relative paths 
        :param transform: Optional transforms (used less frequently for test data).
        """
        self.config = config
        self.dataset_json = config.test_dataset_json
        self.base_dir = config.preprocessed_dir or os.path.dirname(self.dataset_json)
        with open(self.dataset_json, 'r') as f:
            self.data_info = json.load(f)["test"] # Load the test set information

        self.transform = transform


    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, idx):
        # Get the item info
        item_info = self.data_info[idx]

        # Construct full paths for images and labels
        image_path = os.path.join(self.base_dir, item_info["image"])
        label_path = os.path.join(self.base_dir, item_info["label"])

        # Load the image and label from NIfTI files
        image, affine = load_nifti_file(image_path)
        label, _ = load_nifti_file(label_path)

        # Convert to torch tensor (Optional: add channel dim if needed)
        image = torch.from_numpy(image).float().unsqueeze(0) # Add channel dimension
        label = torch.from_numpy(label).long().unsqueeze(0)

        # Apply transforms if any
        if self.transform:
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image), 
                label=tio.LabelMap(tensor=label)
            )
            transformed = self.transform(subject)
            image, label = transformed.image.tensor, transformed.label.tensor

        return image, label
    

# Augmentation and DataLoader Functions

def get_augmentation_transform():
    """
    Returns a composed transform that applies various augmentations to the medical images.
    """
    return tio.Compose([
        tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),  # Random flips along x, y, and z axes
        tio.RandomAffine(scales=(0.9, 1.1), degrees=(0, 20), translation=(0, 10)),  # Random affine transformations
        tio.RandomElasticDeformation(num_control_points=12, max_displacement=(1, 7, 7), locked_borders=2),  # Elastic deformation
        tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.5),  # Random gamma correction (adjust exposure)
        tio.RandomBiasField(p=0.5),  # Bias field to simulate scanner intensity bias
        tio.RandomNoise(mean=0, std=(0, 0.25), p=0.5),  # Gaussian noise
        tio.RescaleIntensity(out_min_max=(0, 1), p=0.5)  # Rescale intensity (replaces contrast adjustment)
    ])


# Dataloader functions for training/validation patches
def get_patch_dataloader(config, shuffle=True, num_workers=4, train=True):
    """
    Create Dataloader for pancreas segmentation patches (train/val).
    """
    augment_transform = get_augmentation_transform() if config.augmented_samples > 1 else None
    dataset = PancreasPatchDataset(config=config,
                                   transform=augment_transform,
                                   train=train)

    if config.distributed:
        sampler = DistributedSampler(dataset)
        shuffle = False # Shuffling is handled by the sampler
    else:
        sampler = None

    dataloader = SubjectsLoader(dataset, 
                            batch_size=config.batch_size, 
                            shuffle=shuffle, 
                            num_workers=num_workers, 
                            pin_memory=True, 
                            sampler=sampler)
    return dataloader


# DataLoader function for test scans
def get_test_dataloader(config,transform=None, num_workers=2):
    """
    Create DataLoader for pancreas segmentation test scans (full NifTI volumes).
    """
    dataset = PancreasTestDataset(config=config, 
                                  transform=transform)
    dataloader = DataLoader(dataset, 
                            batch_size=config.test_batch_size, 
                            shuffle=False, 
                            num_workers=num_workers, 
                            pin_memory=True)
    return dataloader



if __name__ == '__main__':
    import sys
    import os

    # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.config import Config  # Assuming config.py is in the right place
    
    # Initialize the configuration
    config = Config()
    config.augmented_samples = 3  # Set the number of augmented samples to 3 for testing
    config.batch_size = 2  # Testing with a small batch size

    # Get the DataLoader for the training phase
    train_loader = get_patch_dataloader(config=config, train=True)

    # Check the first batch
    for batch_idx, batch in enumerate(train_loader):
        print(f"Batch {batch_idx+1}")
        
        images = batch['image']['data']  # Access the image tensor data
        labels = batch['label']['data']  # Access the label tensor data
        
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image pixel range: [{images.min().item()}, {images.max().item()}]")
        print(f"Label unique values: {torch.unique(labels)}")

        if batch_idx == 4:
            break