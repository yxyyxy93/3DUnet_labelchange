import os


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


if __name__ == "__main__":
    # Set mode for testing
    os.environ['MODE'] = 'train'
    import config
    main()
