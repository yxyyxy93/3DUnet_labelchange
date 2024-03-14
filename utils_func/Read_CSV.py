import numpy as np
import base64


def read_csv_to_3d_array(filepath):
    with open(filepath, 'r') as file:
        # Read the first line to get the dimensions for x and y
        x, y = map(int, file.readline().strip().split(',')[:2])
        # Initialize a list to store 2D slices
        data_slices = []
        for _ in range(x):
            slice_2d = []
            for _ in range(y):
                line = file.readline().strip().split(',')
                # Convert the line to a NumPy array and append to the 2D slice
                slice_2d.append(np.array(line, dtype=float))

            # Stack the 2D slices to form a 3D array
            data_slices.append(np.stack(slice_2d, axis=0))
        # Stack all 2D slices to create the final 3D array
        data_3d_np = np.stack(data_slices, axis=0)

    return data_3d_np


def save_3d_array_to_csv(data_3d_np, filepath):
    """
    Save a 3D NumPy array to a CSV file.

    Args:
        data_3d_np (np.ndarray): The 3D array to be saved.
        filepath (str): The path to the CSV file where the data will be saved.
    """
    with open(filepath, 'w') as file:
        # Write the dimensions of the 3D array as the first line
        x, y, z = data_3d_np.shape
        file.write(f"{x},{y},{z}\n")

        # Write the rest of the data
        for i in range(x):
            for j in range(y):
                ## Convert the slice of the 3D array to a comma-separated string
                line = ','.join([f"{val:.2f}" for val in data_3d_np[i, j, :]])
                file.write(line + '\n')


if __name__ == "__main__":
    # Create a sample 3D array
    sample_data = np.random.rand(5, 5, 3)

    # Define file paths
    save_path = "../dataset/test/exp_test_results.csv"
    load_path = "../dataset/test/exp_test_results.csv"

    # Save to CSV
    save_3d_array_to_csv(sample_data, save_path)
    # Read from CSV
    loaded_data = read_csv_to_3d_array(load_path)

    # Verify if the data matches
    if np.array_equal(sample_data, loaded_data):
        print("Success: The saved and loaded data are identical.")
    else:
        print("Error: The data does not match.")
