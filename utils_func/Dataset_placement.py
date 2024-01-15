import shutil
import os
import re
from collections import defaultdict


def process_file(dirpath, file, pattern, destination):
    """Process file for copying based on parent folder pattern."""
    path_parts = dirpath.split(os.sep)

    # Look for the specific pattern in the path parts
    for part in path_parts:
        if pattern.search(part):
            specific_folder = part
            break
    else:
        # If no matching part is found, return
        return

    # Create new destination path including the specific folder
    new_destination = os.path.join(destination, specific_folder)
    # Ensure this new destination path exists, create if not
    if not os.path.exists(new_destination):
        os.makedirs(new_destination)

    # Full path of the file to be copied
    full_file_path = os.path.join(dirpath, file)

    # Copy the file to the new destination directory
    shutil.copy(full_file_path, new_destination)
    print(f"Copied: {full_file_path} to {new_destination}")


def move_subfolder(src, dest):
    """Move a subfolder from src to dest."""
    # Ensure the destination directory exists
    os.makedirs(dest, exist_ok=True)

    # Move the folder
    shutil.move(src, dest)
    print(f"Moved {src} to {dest}")


if __name__ == "__main__":
    # Define the source directory where to begin the search
    source_directory = r'D:\pogo_work\results_mat\test_woven_[#090]8_(0-1defect)'

    # Define the file types you are looking for
    filetypes = ['origin.csv', 'amplitude.csv', 'phase.csv']

    # Define the destination directory where files will be copied
    destination_directory = r'..\dataset'

    # Define the pattern for the parent folder name you're interested in
    parent_folder_pattern = re.compile(r'seed')  # Pattern to match 'seed' in folder names

    # Ensure the destination directory exists, create if not
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Dictionaries to store files
    files_dict = defaultdict(list)

    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(source_directory):
        for file in filenames:
            if any(file.endswith(filetype) for filetype in filetypes):
                files_dict['sim_data'].append((dirpath, file))
            elif file.startswith('structure'):
                files_dict['sim_struct'].append((dirpath, file))

    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(source_directory):
        for file in filenames:
            # Check if the file ends with any of the specified filetypes
            if any(file.endswith(filetype) for filetype in filetypes):
                # Process for 'seed' in parent folder
                process_file(dirpath, file, parent_folder_pattern,
                             os.path.join(destination_directory, 'sim_data'))

            # Additional condition to check if the file starts with 'structure'
            elif file.startswith('structure'):
                # Process for files starting with 'structure'
                # Here 'structure' is used as the folder name, but you can change it as needed
                process_file(dirpath, file, parent_folder_pattern,
                             os.path.join(destination_directory, 'sim_struct'))

    print("File copying complete.")

    for i in range(128, 149):  # 149 because the upper limit in range is exclusive
        move_subfolder(fr'..\dataset\sim_data\base_model_shiftseed_{i}', r'..\dataset\test\sim_data')
        move_subfolder(fr'..\dataset\sim_struct\base_model_shiftseed_{i}', r'..\dataset\test\sim_struct')