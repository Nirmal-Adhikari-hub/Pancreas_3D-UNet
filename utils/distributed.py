import math
import numpy as np

def get_patch_slices(volume_shape, depth, overlap):
    """
    Get the slices for extracting overlapping patches along the depth axis from a 3D volume.
    :param volume_shape: Tuple (D, H, W) indicating the dimensions of the input volume.
    :param depth: Depth of each patch.
    :param overlap: Overlap between patches along the depth axis.
    :return: List of tuples, each containing slice objects for patch extraction.
    """
    d, h, w = volume_shape  # We are only interested in the depth axis, so h and w are ignored for slicing.
    # print(f"Volume Shape: {volume_shape}")
    
    step_d = depth - overlap  # Step size in depth direction
    patches_slices = []

    # Generate slices along the depth axis, adjusting for the final patch to always have depth = 32
    for z in range(0, d, step_d):
        end_z = min(z + depth, d)  # Ensure the depth doesn't exceed the volume
        if end_z - z < depth:  # If we can't get a full 32-depth patch
            z = d - depth  # Move the starting point to ensure the last patch has depth of 32
            end_z = d  # Adjust the end to fit exactly to the end of the volume
        patches_slices.append((slice(z, end_z), slice(0, h), slice(0, w)))
        if end_z == d:  # Stop when we reach the end of the depth
            break

    return patches_slices


def pad_if_needed(volume, target_depth):
    """ Pad the volume if the depth is smaller than the target depth. """
    d, h, w = volume.shape
    if d < target_depth:
        pad_size = target_depth - d
        padding = ((0, pad_size), (0, 0), (0, 0))  # Pad depth dimension only
        volume = np.pad(volume, padding, mode='constant', constant_values=0)
    return volume
