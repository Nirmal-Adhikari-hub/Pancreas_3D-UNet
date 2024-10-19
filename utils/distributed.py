import math

def get_patch_slices(volume_shape, depth, height, width, overlap):
    """
    Get the slices for extracting overlapping patches from a 3D volume.
    :param volume_shape: Tuple (D, H, W) indicating the dimensions of the input volume.
    :param depth: Depth of each patch.
    :param height: Height of each patch.
    :param width: Width of each patch.
    :param overlap: Overlap between patches.
    :return: List of tuples, each containing slice objects for patch extraction.
    """
    d, h, w = volume_shape

    # Compute the step size (distance between patch starts)
    step_d = depth - overlap
    step_h = height - overlap
    step_w = width - overlap

    # Ensure we don't step beyond the volume boundaries
    patches_slices = []
    for z in range(0, d - depth + 1, step_d):
        for y in range(0, h - height + 1, step_h):
            for x in range(0, w - width + 1, step_w):
                patch_slice = (
                    slice(z, z + depth),
                    slice(y, y + height),
                    slice(x, x + width)
                )
                patches_slices.append(patch_slice)

    return patches_slices
