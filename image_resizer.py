from PIL import Image
import os
from tqdm import tqdm

# Define the folder containing the images and the resized folder
input_folder = 'data/UNBC/img'
output_folder = 'data/UNBC/resized_img'

# Walk through all directories and files in the input folder
for root, dirs, files in os.walk(input_folder):
    # Calculate the relative path to maintain the directory structure
    relative_path = os.path.relpath(root, input_folder)
    # Create the corresponding subfolder in the output folder
    target_subfolder = os.path.join(output_folder, relative_path)
    os.makedirs(target_subfolder, exist_ok=True)

    # Loop through all files in the current folder
    for filename in tqdm(files, desc=f"Processing Images in {relative_path}", leave=False):
        input_path = os.path.join(root, filename)

        # Check if the file is a valid image by extension
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            try:
                # Open the image
                img = Image.open(input_path)

                # Ensure the image size is 172x172 before resizing
                if img.size == (172, 172):
                    # Resize the image to 224x224
                    img_resized = img.resize((224, 224))
                    # Save the resized image to the corresponding output subfolder
                    output_path = os.path.join(target_subfolder, filename)
                    img_resized.save(output_path)
                else:
                    # Log if the image size doesn't match
                    tqdm.write(f"Skipping {filename} in {relative_path}: Image size is not 172x172.")
            except Exception as e:
                # Log any processing errors
                tqdm.write(f"Error processing {filename} in {relative_path}: {e}")
        else:
            # Log if the file type is invalid
            tqdm.write(f"Skipping {filename} in {relative_path}: Not a valid image file.")