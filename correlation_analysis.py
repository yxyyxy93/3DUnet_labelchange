from tqdm import tqdm

from test import SimpleCSVLoader
from utils_func.Read_CSV import read_csv_to_3d_array

if __name__ == "__main__":
    # Initialize model
    import numpy as np
    import os
    import csv

    # Set mode for testing
    os.environ['MODE'] = 'test'

    # Parameters
    fold_number = 1
    test_data_path = "dataset/test/_snr_100000.00_Inst_amplitude_090_1.csv"
    sim_data_path = "D:/pogo_work/results_mat/test_woven_[#090]8_(0-1defect)"
    process_from_start = True  # User-defined flag to choose processing mode
    step = 15

    # Load and preprocess test data
    testdata = SimpleCSVLoader(test_data_path)
    testdata.load_and_preprocess()
    segment_data, original_size = testdata.segment_dataset(chunk_size=(16, 16), step=step)

    # flatten the 3D data
    flattened_data_list = []
    # Loop through each 4D array in the list
    for data in segment_data:
        # Extract the first row of the first dimension for each item
        extracted_data = data[0, 0::4, :, :]
        # Flatten the extracted data
        flattened_data = extracted_data.flatten()
        # Store the flattened data in the new list
        flattened_data_list.append(flattened_data)

    print(f"Image data size: {extracted_data.shape}")

    # Base directory path
    all_files = []
    # First, collect all the relevant files to process
    for root, dirs, files in os.walk(sim_data_path):
        path_parts = root.split(os.sep)
        if len(path_parts) > len(sim_data_path.split(os.sep)) + 1:
            for filename in files:
                full_file_path = os.path.join(root, filename)
                all_files.append(full_file_path)

    # Initialize a list to store results
    results = []
    for full_file_path in tqdm(all_files, desc="Processing files"):
        try:
            image_data = read_csv_to_3d_array(full_file_path)
            image_data = np.transpose(image_data, (2, 0, 1))
            # Segmenting logic
            target_shape = (16, 16)
            start_h = image_data.shape[1] // 2 - target_shape[0] // 2
            start_w = image_data.shape[2] // 2 - target_shape[1] // 2
            end_h = start_h + target_shape[0]
            end_w = start_w + target_shape[1]
            image_data = image_data[0::4, start_h:end_h, start_w:end_w]
            # Initialize a list for correlation coefficients
            corr_coeffs = []
            for exp_data in flattened_data_list[::100]:
                flattened_image_data = image_data.flatten()
                corr_matrix = np.corrcoef(exp_data/max(exp_data), flattened_image_data/max(flattened_image_data))
                corr_coeff = corr_matrix[0, 1]
                corr_coeffs.append(corr_coeff)
            # Calculate the average correlation coefficient
            average_corr_coeff = np.mean(corr_coeffs)
            # Append the path and average correlation coefficient to the results list
            results.append((full_file_path, average_corr_coeff))
        except Exception as e:
            print(f"Error reading or processing file {os.path.basename(full_file_path)}: {e}")

    # After processing all files, save the results to a CSV file
    results_file_path = 'correlation_results.csv'
    with open(results_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['File Path', 'Average Correlation Coefficient'])
        writer.writerows(results)

    print(f"Results saved to {results_file_path}")
