import cv2
import numpy as np
import os

# Set the input and output directories
input_dir = "./input/"
output_dir = "./remove_bg/"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through each subdirectory in the input directory
for sub_dir in os.listdir(input_dir):
    sub_dir_path = os.path.join(input_dir, sub_dir)

    # Check if the item is a directory
    if os.path.isdir(sub_dir_path):
        # Create a corresponding directory in the output folder
        output_sub_dir = os.path.join(output_dir, sub_dir)
        os.makedirs(output_sub_dir, exist_ok=True)

        # Process each .tif file in the subdirectory
        for filename in os.listdir(sub_dir_path):
            if filename.endswith(".tif"):
                # Read the image
                filepath = os.path.join(sub_dir_path, filename)
                imgo = cv2.imread(filepath)

                # Get image dimensions
                height, width = imgo.shape[:2]

                # Create a mask holder
                mask = np.zeros(imgo.shape[:2], np.uint8)

                # Initialize models for GrabCut
                bgdModel = np.zeros((1, 65), np.float64)
                fgdModel = np.zeros((1, 65), np.float64)

                # Define the rectangle for GrabCut
                rect = (10, 10, width - 30, height - 30)

                # Apply GrabCut to remove background
                cv2.grabCut(
                    imgo, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT
                )
                mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
                img1 = imgo * mask[:, :, np.newaxis]

                # Get the background
                background = cv2.absdiff(imgo, img1)

                # Change all non-black background pixels to white
                background[np.where((background > [0, 0, 0]).all(axis=2))] = [
                    255,
                    255,
                    255,
                ]

                # Combine the foreground with the white background
                final = background + img1

                # Save the final output image in the output directory with the same structure
                output_path = os.path.join(output_sub_dir, filename)
                cv2.imwrite(output_path, final)
                print(f"Processed and saved: {output_path}")

print("All images processed.")
