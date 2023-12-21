import os

def remove_duplicate_images(folder1, folder2):
    # Get the names (without file extensions) of all files in the first folder.
    names_folder1 = {os.path.splitext(filename)[0] for filename in os.listdir(folder1)}

    # Traverse the files in the second folder.
    for filename in os.listdir(folder2):
        # Get the file name without the file extension.
        name_without_extension = os.path.splitext(filename)[0]

        # If a file with the same name exists in the first folder, delete that file from the second folder.
        if name_without_extension in names_folder1:
            os.remove(os.path.join(folder2, filename))
            print(f"Removed {filename} from {folder2}")

folder1 = 'VOCdevkit\VOC2007/train3'  # Path to the first folder.
folder2 = 'img'  # Path to the second folder.
remove_duplicate_images(folder1, folder2)
