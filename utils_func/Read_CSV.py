import numpy as np


def read_csv_to_3d_array(filepath):
    # Open the file
    with open(filepath, 'r') as file:
        # Read the first line to get the dimensions
        x, y, z = map(int, file.readline().strip().split(','))

        # Initialize an empty 3D NumPy array
        data_3d_np = np.zeros((x, y, z))

        # Read the rest of the lines and fill the 3D array
        for i in range(x):
            for j in range(y):
                line = file.readline().strip().split(',')
                data_3d_np[i, j, :] = np.array(line[0:z], dtype=float)

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
                # Convert the slice of the 3D array to a comma-separated string
                line = ','.join(map(str, data_3d_np[i, j, :]))
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