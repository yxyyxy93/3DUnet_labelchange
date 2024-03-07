#!/bin/bash

# Define the parent directory to search in
parent_dir="/mnt/raid5/xiaoyu/Ultrasound_data/dataset_woven_[#090]8_0-1defect"

# Loop through all subfolders
find "$parent_dir" -type d | while read -r dir
do
  # Loop through all files in the subfolder
  find "$dir" -type f | while read -r file
  do
    # Extract the filename without the path
    filename=$(basename -- "$file")

    # Loop through numbers 1 to 19
    for i in {1..19}
    do
      # Check if the filename contains the number
      if [[ "$filename" == *"$i"* ]]; then
        # Delete the file
        rm "$file"
        echo "Deleted $file"
        # Break the loop once a file is deleted to avoid trying to delete it multiple times
        break
      fi
    done
  done
done
