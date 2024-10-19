import json
import nibabel as nib
import os
import math

# Path to the dataset JSON file
json_file_path = "../Task07_Pancreas/Task07_Pancreas/dataset.json"
# Path to the imagesTr folder
image_folder = "../Task07_Pancreas/Task07_Pancreas/imagesTr"

def load_json(json_file):
    """
    Loads the dataset.json file and returns its content.
    
    :param json_file: The path to the dataset JSON file.
    :return: The content of the JSON file as a dictionary.
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None

def get_depths_from_json(json_data, image_folder):
    """
    Loops through all the files listed in the JSON data, loads each one, and returns its depth.
    
    :param json_data: The dictionary loaded from the dataset JSON file.
    :param image_folder: Path to the folder containing the NIfTI images.
    :return: A dictionary where the keys are filenames and values are depths.
    """
    depth_list = []
    try:
        # Loop through each training image in the JSON file
        for image_info in json_data['training']:
            image_filename = os.path.basename(image_info['image'])  # Extract the filename
            image_path = os.path.join(image_folder, image_filename)  # Full path to the image

            # Load the NIfTI file and get the depth (Z-axis)
            img = nib.load(image_path)
            data = img.get_fdata()
            depth = data.shape[2]  # The Z-axis dimension
            
            # Store the depth in a dictionary
            depth_list.append(depth)
            print(f"Depth of the file '{image_filename}': {depth}")

        return depth_list
    
    except Exception as e:
        print(f"Error processing files from JSON: {e}")
        return None

if __name__ == "__main__":
    # Load the dataset.json file
    json_data = load_json(json_file_path) and None

    # If JSON data loaded successfully, get the depths of all images
    if json_data:
        depths = get_depths_from_json(json_data, image_folder)

        # Optionally, print the total number of files processed
        print(f"\nTotal files processed: {len(depths)}")
        print(f"\n\nDepths: {depths}")

    depths = [97, 126, 84, 42, 121, 60, 111, 73, 75, 90, 95, 57, 97, 75, 85, 89, 86, 105, 99, 95, 113, 87, 92, 47, 81, 89, 
              116, 89, 88, 93, 97, 113, 83, 68, 161, 93, 81, 104, 71, 89, 57, 169, 84, 44, 123, 76, 91, 83, 81, 44, 72, 85, 
              63, 71, 97, 44, 99, 91, 66, 93, 75, 57, 102, 103, 100, 106, 103, 98, 110, 95, 89, 77, 96, 95, 48, 134, 94, 77, 
              93, 99, 101, 109, 113, 97, 109, 55, 105, 99, 94, 81, 98, 104, 110, 76, 95, 96, 134, 99, 37, 43, 93, 57, 51, 85, 
              85, 87, 85, 85, 93, 102, 103, 93, 89, 59, 81, 93, 45, 56, 97, 91, 51, 86, 164, 91, 73, 107, 88, 104, 89, 93, 73, 
              384, 80, 113, 105, 117, 89, 98, 95, 87, 79, 101, 81, 89, 93, 81, 92, 751, 89, 79, 103, 107, 95, 95, 91, 131, 87, 
              76, 83, 76, 48, 92, 93, 98, 174, 81, 89, 73, 97, 95, 93, 41, 107, 87, 77, 77, 87, 97, 98, 83, 96, 95, 85, 119, 82, 
              107, 105, 90, 105, 92, 61, 112, 103, 99, 89, 89, 84, 84, 79, 89, 44, 40, 106, 113, 93, 87, 121, 39, 118, 96, 104, 
              101, 99, 89, 113, 49, 103, 101, 89, 107, 100, 91, 121, 85, 109, 89, 67, 93, 99, 109, 89, 192, 81, 147, 111, 51, 93, 
              51, 101, 103, 95, 98, 84, 100, 93, 73, 97, 121, 107, 137, 115, 95, 104, 117, 92, 108, 117, 85, 101, 103, 82, 97, 120, 
              89, 98, 130, 50, 83, 58, 142, 97, 105, 137, 87, 97, 98, 81, 101, 55, 85, 87]
    
    print(f"Max Depth: {max(depths)} \t Min Depth: {min(depths)} \t Average: {sum(depths) / len(depths)}")
    print(len([d for d in depths if d <= 256 and d >= 128]))
    print(len([97, 126, 84, 42, 121, 60, 111, 73, 75, 90, 95, 57, 97, 75, 85, 89, 86, 105, 99, 95, 113, 87, 92, 47, 81, 89, 
              116, 89, 88, 93, 97, 113, 83, 68, 161, 93, 81, 104, 71, 89, 57, 169, 84, 44, 123, 76, 91, 83, 81, 44, 72, 85, 
              63, 71, 97, 44, 99, 91, 66, 93, 75, 57, 102, 103, 100, 106, 103, 98, 110, 95, 89, 77, 96, 95, 48, 134, 94, 77, 
              93, 99, 101, 109, 113, 97, 109, 55, 105, 99, 94, 81, 98, 104, 110, 76, 95, 96, 134, 99, 37, 43, 93, 57, 51, 85, 
              85, 87, 85, 85, 93, 102, 103, 93, 89, 59, 81, 93, 45, 56, 97, 91, 51, 86, 164, 91, 73, 107, 88, 104, 89, 93, 73, 
              384, 80, 113, 105, 117, 89, 98, 95, 87, 79, 101, 81, 89, 93, 81, 92, 751]))
