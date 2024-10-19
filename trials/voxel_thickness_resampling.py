import nibabel as nib
import numpy as np
import os
import json
from scipy.ndimage import zoom

# Path to your dataset JSON file and images folder
json_file_path = "../Task07_Pancreas/Task07_Pancreas/dataset.json"
image_folder = "../Task07_Pancreas/Task07_Pancreas/imagesTr"

def load_json(json_file):
    """
    Load the dataset.json file and return its content.
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None

def resample_ct_scan(nifti_file, target_voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Resample the NIfTI file to the target voxel spacing (X, Y, Z).
    
    :param nifti_file: Path to the NIfTI file to be resampled.
    :param target_voxel_spacing: Desired voxel spacing (X, Y, Z).
    :return: Resampled 3D volume and new affine matrix.
    """
    try:
        # Load the NIfTI file
        img = nib.load(nifti_file)
        data = img.get_fdata()  # Get the 3D image data
        header = img.header  # Get the header information

        print(f"Before sampling: {data.shape}")
        
        # Get the current voxel spacing (X, Y, Z)
        current_voxel_spacing = header.get_zooms()[:3]
        print(f"Original Voxel Spacing (X, Y, Z): {current_voxel_spacing}")
        
        # Calculate the zoom factor for each dimension
        zoom_factors = [current / target for current, target in zip(current_voxel_spacing, target_voxel_spacing)]
        print(f"Zoom Factors: {zoom_factors}")
        
        # Resample the data using the zoom factors
        resampled_data = zoom(data, zoom_factors, order=1)  # Linear interpolation
        
        # Adjust the affine matrix (spatial transformation matrix)
        new_affine = np.copy(img.affine)
        new_affine[:3, :3] = np.diag(target_voxel_spacing)  # Update the voxel spacing in the affine matrix
        
        print(f"Resampled Shape: {resampled_data.shape}")
        return resampled_data, new_affine
    
    except Exception as e:
        print(f"Error in resampling: {e}")
        return None, None

def resample_single_file(json_data, image_folder, target_voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Load the first CT scan file from the JSON, resample it, and return the resampled data.
    
    :param json_data: JSON data loaded from the dataset.json file.
    :param image_folder: Folder where NIfTI files are stored.
    :param target_voxel_spacing: Desired voxel spacing for resampling.
    :return: None
    """
    try:
        # Get the first image from the training set in the JSON file
        first_image_info = json_data['training'][147]  # Get the first CT scan info
        image_filename = os.path.basename(first_image_info['image'])
        nifti_file_path = os.path.join(image_folder, image_filename)
        
        print(f"Resampling file: {nifti_file_path}")
        
        # Resample the CT scan to the target voxel spacing
        resampled_data, new_affine = resample_ct_scan(nifti_file_path, target_voxel_spacing=target_voxel_spacing)
        
        if resampled_data is not None:
            print("Resampling completed.")
            # Optionally save the resampled NIfTI file
            resampled_img = nib.Nifti1Image(resampled_data, affine=new_affine)
            nib.save(resampled_img, f"resampled_{image_filename}")
            print(f"Resampled NIfTI file saved as 'resampled_{image_filename}'")
        
    except Exception as e:
        print(f"Error in processing: {e}")

if __name__ == "__main__":
    # Load the JSON data
    json_data = load_json(json_file_path)
    
    if json_data:
        # Resample the first file to a target voxel spacing of 1.0mm in each direction (X, Y, Z)
        resample_single_file(json_data, image_folder, target_voxel_spacing=(1.0, 1.0, 1.0))
