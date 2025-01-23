import os


def rename_images(directory_path):
    """
    Renames all JPEG images in the specified directory to have sequential filenames starting from 1.
    """
    # Get a list of all files in the directory
    files = os.listdir(directory_path)

    # Filter only JPEG images
    jpg_files = [file for file in files if file.lower().endswith('.jpg')]

    # Sort the list of images
    jpg_files.sort()

    # Rename the images sequentially
    for i, file_name in enumerate(jpg_files, start=1):
        # Construct the new file name
        new_file_name = f"{i}.jpg"

        # Construct the full paths of the old and new files
        old_file_path = os.path.join(directory_path, file_name)
        new_file_path = os.path.join(directory_path, new_file_name)

        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed '{file_name}' to '{new_file_name}'")


# Example usage:
directory_path = '/home/thales1/ODS4kkaggle/samples'  # Change this to your directory path
rename_images(directory_path)
