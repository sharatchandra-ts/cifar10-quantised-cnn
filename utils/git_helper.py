import random

import numpy as np

from data_loader import get_cifar10_loaders

# 1. Grab the loaders
trainloader, valloader, testloader = get_cifar10_loaders(greyscale=True)

# 2. Extract the random raw sample
underlying_dataset = trainloader.dataset.dataset
random_idx = 69

# Grab the raw image data (32x32x3) and its corresponding integer label
raw_img = underlying_dataset.data[random_idx]
image_label = underlying_dataset.targets[random_idx]

# Map the integer label to its class name (CIFAR-10 classes)
classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
label_name = classes[image_label]

print(f"Selected random image index: {random_idx}")
print(f"Image Label ID: {image_label} -> Class Name: {label_name}")
print(f"Original array shape: {raw_img.shape}, Data Type: {raw_img.dtype}")

# 3. Convert to Grayscale manually if needed
if len(raw_img.shape) == 3:
    raw_img = (
        0.299 * raw_img[:, :, 0] + 0.587 * raw_img[:, :, 1] + 0.114 * raw_img[:, :, 2]
    ).astype(np.uint8)

print(f"Processed grayscale shape: {raw_img.shape}")  # Should be (32, 32)

# ─── SAVE TO HEX FILE (ROW-BY-ROW FORMAT) ──────────────────────────────────
# Each line in the hex file will be one entire row (32 pixels * 2 hex chars = 64 hex chars)

with open("image_rows.hex", "w") as f:
    for row in raw_img:
        # Convert each pixel in the row to a 2-character hex string
        # and join them together into one 64-character long string
        row_hex = "".join(f"{pixel:02x}" for pixel in row)
        f.write(f"{row_hex}\n")

print(
    "Saved 'image_rows.hex' (32 lines, each line is 64 hex characters / 256-bits wide)"
)


import matplotlib.pyplot as plt
import numpy as np


def visualize_hex_image(hex_file_path):
    try:
        pixel_grid = []

        with open(hex_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Each line is 64 hex characters (32 pixels * 2 chars)
                # Split the line into 2-character pairs
                row_pixels = [int(line[i : i + 2], 16) for i in range(0, len(line), 2)]
                pixel_grid.append(row_pixels)

        # Convert the list of lists into a 2D NumPy array
        img_array = np.array(pixel_grid, dtype=np.uint8)

        print(f"Reconstructed Image Shape: {img_array.shape}")

        # Plot the image using matplotlib
        plt.figure(figsize=(4, 4))
        plt.imshow(img_array, cmap="gray", vmin=0, vmax=255)
        plt.title(f"Visualized Grid ({hex_file_path})")
        plt.axis("off")  # Hide pixel coordinate axes
        plt.show()

    except FileNotFoundError:
        print(
            f"Error: The file '{hex_file_path}' was not found. Make sure you generated it first!"
        )
    except ValueError as e:
        print(
            f"Error parsing hex values: {e}. Check if the file formatting is correct."
        )


# Run the visualizer on your file
visualize_hex_image("image_rows.hex")
