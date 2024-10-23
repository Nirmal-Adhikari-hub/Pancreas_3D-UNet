import os
import numpy as np
import json
import nibabel as nib
from sklearn.model_selection import train_test_split
import sys
from scipy.ndimage import zoom # For resampling


# Logger class to log stdout and stderr to both terminal and file
class TeeLogger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = open(log_file, "w", buffering=1)

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

# Add parent directory to the Python path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.distributed import get_patch_slices, pad_if_needed


class PreprocessPancreasDataset:
    def __init__(self, config, output_dir="/shared/home/xvoice/nirmal/data/Task07_Pancreas/Preprocessed"):
        self.config = config
        self.output_dir = output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        #Load dataset info (dataset.json)
        with open(config.dataset_json, 'r') as f:
            dataset_info = json.load(f)

        # Full training data paths
        self.data_info = dataset_info['training']

        # Step 1: Split into train, val and test set (CT scan levels)
        self.train_data, self.test_data = train_test_split(self.data_info, test_size=0.1, random_state=42)

        # Split the train data into train/val split
        self.train_data, self.val_data = train_test_split(self.train_data, test_size=0.1, random_state=42)

        print(f"Train Size: {len(self.train_data)}, Val Size: {len(self.val_data)}, Test Size: {len(self.test_data)}")

        # Prepare the directories to store the paths of preprocessed patches and labels
        self.preprocessed_data = {
            "train": [],
            "val": [],
            "test": []
        }


    def resample_volume(self, image, voxel_spacing, target_spacing=[1,1,1], is_label=False):
        """
        Resample the image based on the provided voxel spacing.
        Only resampling along the z-axis.
        """
        zoom_factors = np.array(voxel_spacing) / np.array(target_spacing)
        # Use the nearest neighbour interpolation for labels (0) and linear for image(1)
        order = 0 if is_label else 1
        # Resample the volume
        resampled_image = zoom(image, (zoom_factors[2], 1, 1), order=order)
        print(f"Label Shape {image.shape} resampled to {resampled_image.shape}\n") if is_label else print(f"Image Shape {image.shape} resampled to {resampled_image.shape}\n")
        return resampled_image
    

    def process_and_save(self):
        """
        Process the train and validation sets by slicing patches, saving them, and storing 
        their metadata in a new JSON file.
        The test set is kept as raw volumes.
        """

        # Ensure train, val, and test directories are created
        train_dir = os.path.join(self.output_dir, 'train')
        val_dir = os.path.join(self.output_dir, 'val')
        test_dir = os.path.join(self.output_dir, 'test')

        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        # Process train and validation sets
        for phase, data in zip(['train', 'val'], [self.train_data, self.val_data]):
            for entry in data:
                img_path = os.path.join(self.config.dataset_path, entry['image'])
                label_path = os.path.join(self.config.dataset_path, entry['label'])

                print(f"Preprocessing {os.path.basename(img_path).replace('.nii.gz', '')} for {phase} phase.")

                # Load the NIfTI files
                img_nifti = nib.load(img_path)
                label_nifti = nib.load(label_path)

                # Convert to numpy arrays and ensure correct axes order
                image = np.transpose(np.array(img_nifti.get_fdata(), dtype=np.float32), (2, 0, 1)) # Shape: (D, H, W)
                label = np.transpose(np.array(label_nifti.get_fdata(), dtype=np.uint8), (2, 0, 1)) # Shape: (D, H, W)

                # Voxel Resampling (along the depth axis only)
                voxel_spacing = img_nifti.header.get_zooms()[:3]
                image_resampled = self.resample_volume(image, voxel_spacing)
                label_resampled = self.resample_volume(label, voxel_spacing, is_label=True)

                # Slice the pathces and save them
                patches, labels = self.get_patches(image_resampled, label_resampled)
                self.save_patches(phase, patches, labels, entry['image'])

        # Process the Test set (no slicing, just save the full scans)
        for entry in self.test_data:
            img_path = os.path.join(self.config.dataset_path, entry['image'])
            label_path = os.path.join(self.config.dataset_path, entry['label'])

            print(f"Saving raw scan for test: {img_path}")

            # Load and resample the NIfTI files
            img_nifti = nib.load(img_path)
            label_nifti = nib.load(label_path)

            image = np.transpose(np.array(img_nifti.get_fdata(), dtype=np.float32), (2, 0, 1)) # Shape: (D, H, W)
            label = np.transpose(np.array(label_nifti.get_fdata(), dtype=np.float32), (2, 0, 1)) # Shape: (D, H, W)

            voxel_spacing = img_nifti.header.get_zooms()[:3]
            image_resampled = self.resample_volume(image, voxel_spacing)
            label_resampled = self.resample_volume(label, voxel_spacing, is_label=True)

            # Create destination paths inside the test directory
            image_filename = os.path.basename(img_path)
            label_filename = os.path.basename(label_path).replace('.nii.gz', '_label.nii.gz')
            dest_img_path = os.path.join(test_dir, image_filename)
            dest_label_path = os.path.join(test_dir, label_filename)

            # Save the resampled CT scan files to the test folder
            nib.save(nib.Nifti1Image(image_resampled.transpose(1, 2, 0), img_nifti.affine), dest_img_path)
            nib.save(nib.Nifti1Image(label_resampled.transpose(1, 2, 0), label_nifti.affine), dest_label_path)

            # Store the paths in the preprocessed data
            self.preprocessed_data['test'].append({
                "image": dest_img_path,
                "label": dest_label_path
            })


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
        print(f"Extracted {len(patches)} patches from the CT scan of size {image.shape}")
        return patches, labels
    
    def save_patches(self, phase, patches, labels, image_filename):
        """
        Save the patches and corresponding labels as .npy files to the 
        preprocessed directories.
        If previous patches exist, they are deleted before saving new ones.
        """

        # Get the basename of the image file (eg pancreas_001)
        image_basename = os.path.basename(image_filename).replace('.nii.gz', '')

        # Define the target directory based on the phase (train/val)
        phase_dir = os.path.join(self.output_dir, phase)

        # Ensure the directory exists
        if not os.path.exists(phase_dir):
            os.makedirs(phase_dir)

        # Iterate through the patches and labels and save them as .npy files
        for i, (patch, label) in enumerate(zip(patches, labels)):
            patch_file = f"{image_basename}_patch_{i}.npy"
            label_file = f"{image_basename}_label_{i}.npy"

            # Save patch and label numpy arrays
            np.save(os.path.join(phase_dir, patch_file), patch)
            np.save(os.path.join(phase_dir, label_file), label)

            # Append metadata for the preprocessed data for saving to JSON later
            self.preprocessed_data[phase].append({
                "image": os.path.join(phase_dir, patch_file),
                "label": os.path.join(phase_dir, label_file)
            })

            print(f"Saved patch: {os.path.join(phase_dir, patch_file)} from {image_basename}")
            print(f"Saved label: {os.path.join(phase_dir, label_file)} from {image_basename}")


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

        print(f"Preprocessed dataset JSON files saved: {train_json_path}\n {val_json_path} \n {test_json_path}")


if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.config import Config

    config = Config()

    preprocessor = PreprocessPancreasDataset(config)

    preprocessor.process_and_save()