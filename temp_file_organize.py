import os


def delete_non_csv_files(root_folder):
    # Traverse through the directories
    for subdir, dirs, files in os.walk(root_folder):
        # Check if we're at the subfolder level, not sub-subfolder level
        if subdir.count(os.sep) == root_folder.count(os.sep) + 1:
            for file in files:
                # Check file extension
                if not file.endswith('.csv'):
                    file_path = os.path.join(subdir, file)
                    print(f"Deleting: {file_path}")
                    os.remove(file_path)


# Specify the root folder
root_folder = "D:/pogo_work/results_mat/test_woven_[#090]8_(0-1defect)"

delete_non_csv_files(root_folder)
