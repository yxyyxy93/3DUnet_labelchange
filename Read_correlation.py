import csv
import re
import matplotlib.pyplot as plt
import numpy as np


def extract_numbers_from_path(file_path):
    # Split the file path by "\"
    parts = file_path.split("\\")
    # Initialize variables to hold your extracted numbers
    Start_point, struct_num, snr = None, None, None
    # Use regular expressions to find numbers following specific patterns
    for part in parts:
        if 'amplitude' in part:
            # Look for a number following "amplitude_"
            match = re.search(r'amplitude_(\d+)', part)
            if match:
                Start_point = match.group(1)
        elif 'shiftseed' in part:
            # Look for a number following "shiftseed_"
            match = re.search(r'shiftseed_(\d+)', part)
            if match:
                struct_num = match.group(1)
        elif '_snr_' in part:
            # Look for a number following "_snr_"
            match = re.search(r'_snr_([\d.]+)', part)
            if match:
                snr = match.group(1)
    return Start_point, struct_num, snr


# File containing the results
input_csv = 'correlation_results.csv'

# New matrix to hold the extracted information and correlation coefficients
new_matrix = []

# Read the CSV file
with open(input_csv, 'r') as file:
    reader = csv.reader(file)
    next(reader, None)  # Skip the header row
    data = list(reader)

# Convert the second column to float for sorting
for i in range(len(data)):
    data[i][1] = float(data[i][1])

# Sequence the data based on the Correlation Coefficient
data.sort(key=lambda x: x[1], reverse=True)  # Sort by correlation coefficient, highest first


# Extract information and append to the new matrix
for file_path, corr_coeff in data:
    start_point, struct_num, snr = extract_numbers_from_path(file_path)
    new_matrix.append([start_point, struct_num, snr, corr_coeff])

## ************** read defect info ****************************
# Define the path to your file
file_path = 'defects_log.txt'
# Read the content of the file
with open(file_path, 'r') as file:
    text = file.read()
# Split the text into lines
lines = text.strip().split('\n')
# Initialize a list to store the extracted data
extracted_data = []
# Variables to temporarily store data from the text
current_set_number = None
current_num_defects = None
# Process each line
for line in lines:
    # Check if the line contains 'Set Name'
    if 'Set Name:' in line:
        # Extract the last number in 'Set Name'
        match = re.search(r'shiftseed_(\d+)', line)
        if match:
            current_set_number = int(match.group(1))
    # Check if the line contains 'Num Defects'
    elif 'Num Defects:' in line:
        # Extract the number of defects
        current_num_defects = int(line.split(':')[-1].strip())
        # Once both 'Set Name' and 'Num Defects' are found, append them to the list
        extracted_data.append([int(current_set_number), int(current_num_defects)])
# Convert the list of lists into a 2*n matrix (for demonstration purposes, it remains a list of lists)
matrix_2n = extracted_data

# *************************
# replace struct_num with the defect number
map_set_num_to_defects = {set_num: num_defects for set_num, num_defects in matrix_2n}
print(map_set_num_to_defects)
# replacement operation
for row in new_matrix:
    set_num = int(row[1])  # Assuming the first element is the identifier for the lookup
    if set_num in map_set_num_to_defects:
        # Assuming we are replacing the entire row[1] or a specific value in the row with Num Defects
        num_defects = map_set_num_to_defects[set_num]
        # Here you replace a specific value in the row; adjust according to your actual structure
        row[1] = num_defects  # This directly updates the corresponding element with Num Defects
    else:
        print(f"Set number {set_num} not found in the map. No replacement made for this row.")


# Plotting
fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Create 3 subplots vertically
new_matrix = np.array(new_matrix)

for i in range(3):  # Loop through the first three rows
    unique_values = np.unique(new_matrix[:, i])  # Unique values in the current row
    data_for_boxplot = []
    labels = []  # List to hold labels for x-ticks
    for value in unique_values:
        # Collecting corresponding values from the fourth row for each unique value
        corresponding_values = new_matrix[new_matrix[:, i] == value, 3]
        data_for_boxplot.append(np.array(corresponding_values, dtype=float))
        labels.append(str(value))  # Convert value to string for labeling

    # Box plot for the current row
    axs[i].boxplot(data_for_boxplot, patch_artist=True)
    axs[i].set_xticklabels(labels)  # Set the x-tick labels to the unique values

# Plot 1st row against 4th row
axs[0].set_title('start point vs correlation coefficient')
axs[0].set_ylabel('correlation coefficient')
axs[0].set_xlabel('start point')
# Plot 2nd row against 4th row
axs[1].set_title('defect number vs correlation coefficient')
axs[1].set_ylabel('correlation coefficient')
axs[1].set_xlabel('defect number')
# Plot 3rd row against 4th row
axs[2].set_title('SNR vs correlation coefficient')
axs[2].set_ylabel('correlation coefficient')
axs[2].set_xlabel('SNR (dB)')

# ********** box plot for all corr. coef.
# Creating the box plot for the fourth column
plt.figure(figsize=(8, 6))  # Define figure size
plt.boxplot(np.array(new_matrix[:, 3], dtype=float), patch_artist=True)
# Customizing the plot
plt.title('Box Plot for all')
plt.ylabel('correlation coefficient')

plt.tight_layout()
plt.show()

# Assuming new_matrix is already defined and is a NumPy array
# Extract the fourth column which we are interested in analyzing
column_data = np.array(new_matrix[:, 3], dtype=float)  # Ensure it's of type float
# Calculate the descriptive statistics
min_val = np.min(column_data)
q1_val = np.percentile(column_data, 25)
median_val = np.median(column_data)
q3_val = np.percentile(column_data, 75)
max_val = np.max(column_data)
# Calculate IQR (Interquartile Range)
iqr = q3_val - q1_val
# Identify outliers (values that are 1.5 times IQR below Q1 or above Q3)
lower_fence = q1_val - 1.5 * iqr
upper_fence = q3_val + 1.5 * iqr
outliers = column_data[(column_data < lower_fence) | (column_data > upper_fence)]
# Compile the statistics
statistics = {
    "Minimum": min_val,
    "Q1 (25th percentile)": q1_val,
    "Median": median_val,
    "Q3 (75th percentile)": q3_val,
    "Maximum": max_val,
    "IQR": iqr,
    "Lower Fence": lower_fence,
    "Upper Fence": upper_fence,
    "Outliers": outliers.tolist()  # Converting to list for display purposes
}
print(statistics)

# # Print the updated new_matrix for verification
# for row in new_matrix:
#     print(row)

# # If you need to save this new_matrix to a CSV file
# output_csv = 'extracted_data_with_correlation.csv'
# with open(output_csv, 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Start_point', 'Struct_num', 'SNR', 'Correlation Coefficient'])
#     writer.writerows(new_matrix)
