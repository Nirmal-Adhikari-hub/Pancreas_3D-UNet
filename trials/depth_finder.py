import os
import nibabel as nib
import numpy as np
import json

def calculate_voxel_spacing_statistics(dataset_json, dataset_path):
    voxel_spacings = []
    scan_shapes = {}  # Dictionary to store shapes of the scans

    # Load the dataset information from the JSON file
    with open(dataset_json, 'r') as f:
        dataset_info = json.load(f)

    min_voxel_scan = os.path.basename(os.path.join(dataset_path, dataset_info['training'][0]['image']))
    max_voxel_scan = os.path.basename(os.path.join(dataset_path, dataset_info['training'][0]['image']))
    
    # Process each training image
    for entry in dataset_info['training']:
        image_path = os.path.join(dataset_path, entry['image'])
        print(f"Processing {image_path}...")

        # Load the NIfTI image
        img_nifti = nib.load(image_path)
        
        # Get the voxel spacing (pixdim[1:4] gives spacing along x, y, z)
        voxel_spacing = img_nifti.header['pixdim'][1:4]
        voxel_spacings.append(voxel_spacing)

        # Store the shape of the current scan
        scan_shape = img_nifti.shape
        scan_shapes[os.path.basename(image_path)] = scan_shape

        # Update the min and max voxel scan based on current scan
        if np.array_equal(voxel_spacing, np.min(voxel_spacings, axis=0)):
            min_voxel_scan = os.path.basename(image_path)
        if np.array_equal(voxel_spacing, np.max(voxel_spacings, axis=0)):
            max_voxel_scan = os.path.basename(image_path)

    # Convert to numpy array for easier statistics computation
    voxel_spacings = np.array(voxel_spacings)

    # Calculate statistics
    stats = {
        'min_voxel_spacing': np.min(voxel_spacings, axis=0),
        'max_voxel_spacing': np.max(voxel_spacings, axis=0),
        'mean_voxel_spacing': np.mean(voxel_spacings, axis=0),
        'std_voxel_spacing': np.std(voxel_spacings, axis=0),
        'max_voxel_scan': max_voxel_scan,
        'min_voxel_scan': min_voxel_scan,
        'min_voxel_scan_shape': scan_shapes[min_voxel_scan],
        'max_voxel_scan_shape': scan_shapes[max_voxel_scan],
    }

    return stats, voxel_spacings

if __name__ == "__main__":
    # Path to the dataset JSON file and the base path to the dataset folder
    dataset_json = "D:/Nirmal/pancreas/Task07_Pancreas/Task07_Pancreas/dataset.json"
    dataset_path = "D:/Nirmal/pancreas/Task07_Pancreas/Task07_Pancreas"
    
    # Calculate voxel spacing statistics
    stats, vs = calculate_voxel_spacing_statistics(dataset_json, dataset_path)
    
    # Print out the results
    print(f"Voxel Spacing Statistics:")
    print(f"Min voxel spacing (x, y, z): {stats['min_voxel_spacing']}, Image: {stats['min_voxel_scan']}, Shape: {stats['min_voxel_scan_shape']}")
    print(f"Max voxel spacing (x, y, z): {stats['max_voxel_spacing']}, Image: {stats['max_voxel_scan']}, Shape: {stats['max_voxel_scan_shape']}")
    print(f"Mean voxel spacing (x, y, z): {stats['mean_voxel_spacing']}")
    print(f"Std voxel spacing (x, y, z): {stats['std_voxel_spacing']}")

    print(vs)
