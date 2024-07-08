import os
import dataset
import numpy as np
import shutil

def calculate_ratio_ones_zeros(dataset):
    total_ones = 0
    total_zeros = 0

    for data in dataset:
        gt = data['gt'].numpy()  # Ground truth tensor
        ones = np.sum(gt == 1)
        zeros = np.sum(gt == 0)
        total_ones += ones
        total_zeros += zeros

    ratio = total_ones / total_zeros if total_zeros != 0 else float('inf')
    return ratio, total_ones, total_zeros


def main():
    # Define the paths to the main directories
    dirs1 = config.image_dirs
    dir2 = config.label_dir

    # Gather all subfolders from multiple directories in dirs1
    subfolders1 = set()
    for dir1 in dirs1:
        if os.path.exists(dir1):
            subfolders1.update({folder for folder in os.listdir(dir1) if os.path.isdir(os.path.join(dir1, folder))})
        else:
            print(f"Directory '{dir1}' does not exist.")

    # List subfolders in dir2
    subfolders2 = {folder for folder in os.listdir(dir2) if os.path.isdir(os.path.join(dir2, folder))}

    # Find subfolders in subfolders1 that do not have a corresponding folder in subfolders2
    missing_folders = subfolders1 - subfolders2

    # Print out the missing subfolders
    if missing_folders:
        print("Subfolders in '{}' that do not have a corresponding folder in '{}':".format(", ".join(dirs1), dir2))
        for folder in missing_folders:
            print(folder)
    else:
        print("All subfolders in '{}' have a corresponding folder in '{}'.".format(", ".join(dirs1), dir2))

    # # Create the dataset
    # dataset_samples = dataset.TrainValidImageDataset(
    #     config.image_dirs, config.label_dir, option_type=1, dilation_factors=[3, 0, 0],
    #     max_samples=20000)
    # # Calculate the ratio
    # ratio, total_ones, total_zeros = calculate_ratio_ones_zeros(dataset_samples)
    # print(f"Total ones: {total_ones}")
    # print(f"Total zeros: {total_zeros}")
    # print(f"Ratio of ones to zeros: {ratio}")

    # Define the file path
    file_path = 'defects_log.txt'

    # Read the data from the file
    with open(file_path, 'r') as file:
        data = file.read()

    # Split the data into lines
    lines = data.strip().split('\n')

    # Initialize variables
    set_names_with_zero_defects = []
    current_set_name = None

    # Process each line
    for line in lines:
        if line.startswith('Set Name:'):
            current_set_name = line.split(': ')[1]
        elif line.startswith('Num Defects:'):
            num_defects = int(line.split(': ')[1])
            if num_defects == 0 and current_set_name is not None:
                set_names_with_zero_defects.append(current_set_name)

    # Print the result
    print("Set Names with Num Defects equal to 0:")
    for set_name in set_names_with_zero_defects:
        print(set_name)
        subfolder_name = f"base_model_{set_name}"
        subfolder_path = os.path.join('/mnt/raid5/xiaoyu/Ultrasound_data/dataset_woven_['
                                      '#090]8_0-1defect/sim_data_Inst_amplitude_805', subfolder_name)
        if os.path.exists(subfolder_path):
            shutil.rmtree(subfolder_path)
            print(f"Deleted subfolder: {subfolder_path}")
        else:
            print(f"Subfolder not found: {subfolder_path}")


if __name__ == "__main__":
    # Set mode for testing
    os.environ['MODE'] = 'train'
    import config

    main()
