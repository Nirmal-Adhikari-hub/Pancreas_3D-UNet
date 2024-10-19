import nibabel as nib
import torch

def load_image_label(image_path, label_path):
    # Load image and label using nibabel
    image_nii = nib.load(image_path)
    label_nii = nib.load(label_path)
    
    # Convert to numpy arrays
    image_data = image_nii.get_fdata()
    label_data = label_nii.get_fdata()

    # Convert numpy arrays to torch tensors
    image_tensor = torch.tensor(image_data, dtype=torch.float32)
    label_tensor = torch.tensor(label_data, dtype=torch.long)  # long for integer labels
    
    return image_tensor, label_tensor


import matplotlib.pyplot as plt

def visualize_slice(image_tensor, slice_index):
    slice_data = image_tensor[..., slice_index].numpy()  # Get the specific slice from the tensor
    plt.imshow(slice_data, cmap='gray')  # Display in grayscale (since it's CT scan)
    plt.colorbar(label='Hounsfield Units (HU)')
    plt.title(f"CT Slice at Index {slice_index}")
    plt.show()




# Example usage (replace with the actual paths from dataset.json)
image_path = '../Task07_Pancreas/Task07_Pancreas/imagesTr/pancreas_290.nii.gz'
label_path = '../Task07_Pancreas/Task07_Pancreas/labelsTr/pancreas_290.nii.gz'

image_tensor, label_tensor = load_image_label(image_path, label_path)

# Example usage:
visualize_slice(label_tensor, 50)  # Visualize the 50th slice in the depth (Z-axis)

# Print tensor shapes
print(f"Image Tensor : {image_tensor[:,:,80]}")  # e.g., (512, 512, z)
print(f"Label Tensor : {label_tensor[:,:,80]}")  # e.g., (512, 512, z)
