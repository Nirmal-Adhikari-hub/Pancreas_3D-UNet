import os
import numpy as np
import json
import nibabel as nib  # To load NIfTI files
from sklearn.model_selection import train_test_split
import sys
import logging

# Logger class to log stdout and stderr to both terminal and file
class TeeLogger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = open(log_file, "a")

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


class PreprocessPancreasDataset:
    def __init__(self, config, output_dir="/shared/home/xvoice/nirmal/data/Task07_Pancreas/Preprocessed"):
        self.config = config
        self.output_dir = output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Load dataset info (dataset.json)
        with open(config.dataset_json, 'r') as f:
            dataset_info = json.load(f)
        
        # Full training data paths
        self.data_info = dataset_info['training']

        # Step 1: Split into train, validation, and test sets (CT scan level)
        self.train_data, self.test_data = train_test_split(self.data_info, test_size=0.1, random_state=42)

        # Split train data into train/val split
        self.train_data, self.val_data = train_test_split(self.train_data, test_size=0.1, random_state=42)

        print(f"Train Size: {len(self.train_data)}, Val Size: {len(self.val_data)}, Test Size: {len(self.test_data)}")

        # Prepare dictionaries to store the paths of preprocessed patches and labels
        self.preprocessed_data = {
            "train": [],
            "val": [],
            "test": []
        }

    def process_and_save(self):
        """
        Process the train and validation sets by slicing patches, saving them,
        and storing their metadata in a new JSON file.
        The test set is kept as raw volumes.
        """
        # Process Train and Validation sets
        for phase, data in zip(['train', 'val'], [self.train_data, self.val_data]):
            for entry in data:
                img_path = os.path.join(self.config.dataset_path, entry['image'])
                label_path = os.path.join(self.config.dataset_path, entry['label'])
                
                print(f"Processing {img_path} for {phase} phase.")  # Debugging

                # Load the NIfTI files
                img_nifti = nib.load(img_path)
                label_nifti = nib.load(label_path)

                # Convert to numpy arrays and ensure correct axes order
                image = np.transpose(np.array(img_nifti.get_fdata(), dtype=np.float32), (2, 0, 1))  # Shape: (D, H, W)
                label = np.transpose(np.array(label_nifti.get_fdata(), dtype=np.uint8), (2, 0, 1))  # Shape: (D, H, W)

                # Slice the patches and save them
                patches, labels = self.get_patches(image, label)
                self.save_patches(phase, patches, labels, entry['image'])  # Save them offline

        # Process the Test set (no slicing, just save the full scans)
        for entry in self.test_data:
            img_path = os.path.join(self.config.dataset_path, entry['image'])
            label_path = os.path.join(self.config.dataset_path, entry['label'])

            print(f"Saving {img_path} for test phase.")  # Debugging

            # Copy the raw CT scan file paths to the preprocessed JSON
            self.preprocessed_data['test'].append({
                "image": img_path,
                "label": label_path
            })

        # Write the preprocessed dataset to new JSON files
        self.save_json()

    def get_patches(self, image, label):
        """
        Extract patches from a 3D image and corresponding labels.
        """
        depth, height, width = self.config.input_size
        image = pad_if_needed(image, depth)
        label = pad_if_needed(label, depth)
        patch_slices = get_patch_slices(image.shape, depth, self.config.patch_overlap)
        
        patches, labels = [], []
        for sl in patch_slices:
            patches.append(image[sl].copy())
            labels.append(label[sl].copy())
        print(f"Extracted {len(patches)} patches from the CT scan.")  # Debugging
        return patches, labels

    def save_patches(self, phase, patches, labels, image_name):
        """
        Save the patches and labels, and store their metadata for the preprocessed JSON.
        """
        phase_dir = os.path.join(self.output_dir, phase)
        if not os.path.exists(phase_dir):
            os.makedirs(phase_dir)

        patch_paths = []
        for i, (patch, label_patch) in enumerate(zip(patches, labels)):
            patch_file = f"{image_name}_patch_{i}.npy"
            label_file = f"{image_name}_label_{i}.npy"

            # Save patch and label as .npy files
            np.save(os.path.join(phase_dir, patch_file), patch)
            np.save(os.path.join(phase_dir, label_file), label_patch)

            # Add paths to the preprocessed dataset JSON
            patch_paths.append({
                "patch": os.path.join(phase_dir, patch_file),
                "label": os.path.join(phase_dir, label_file)
            })

        print(f"Saved {len(patches)} patches for {image_name} in {phase} phase.")  # Debugging
        self.preprocessed_data[phase].extend(patch_paths)

    def save_json(self):
        """
        Save the preprocessed dataset information to separate JSON files.
        """
        train_json_path = os.path.join(self.output_dir, "train_dataset_preprocessed.json")
        val_json_path = os.path.join(self.output_dir, "val_dataset_preprocessed.json")
        test_json_path = os.path.join(self.output_dir, "test_dataset_preprocessed.json")

        with open(train_json_path, 'w') as f:
            json.dump({"train": self.preprocessed_data['train']}, f, indent=4)

        with open(val_json_path, 'w') as f:
            json.dump({"val": self.preprocessed_data['val']}, f, indent=4)

        with open(test_json_path, 'w') as f:
            json.dump({"test": self.preprocessed_data['test']}, f, indent=4)

        print(f"Preprocessed dataset JSON files saved.")  # Debugging


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.config import Config  # Assuming you already have your Config class
    from utils.distributed import get_patch_slices, pad_if_needed
    config = Config()

    # Initialize the preprocessor
    preprocessor = PreprocessPancreasDataset(config)

    # Start processing and saving patches
    preprocessor.process_and_save()