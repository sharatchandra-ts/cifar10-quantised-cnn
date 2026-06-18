import numpy as np


def load_hex_file(file_path, expected_size=None):
    """
    Reads a hex file, converts hex strings to integers/floats,
    and reshapes to the expected dimensions.
    """
    try:
        with open(file_path, "r") as f:
            # Read all tokens, ignoring lines that start with comment characters if any
            content = f.read().split()

        # Convert hex strings to integers
        # (Change to np.float32 or scale if your hex represents fixed-point/floating numbers)
        data = np.array([int(val, 16) for val in content], dtype=np.float32)

        if expected_size and data.size != expected_size:
            # If the file contains multiple kernels, we handle reshaping later
            if data.size % expected_size != 0:
                raise ValueError(
                    f"Data size {data.size} in {file_path} does not match expected size/multiples."
                )

        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        exit(1)
    except ValueError as e:
        print(f"Error parsing hex data in '{file_path}': {e}")
        exit(1)


def manual_convolution_2d(image, kernel):
    """
    Performs a valid 2D convolution (no padding, stride=1) using NumPy indexing.
    """
    img_h, img_w = image.shape
    ker_h, ker_w = kernel.shape

    # Calculate output dimensions
    out_h = img_h - ker_h + 1
    out_w = img_w - ker_w + 1

    # Create an empty output array
    output = np.zeros((out_h, out_w))

    # Perform the convolution slide
    for i in range(out_h):
        for j in range(out_w):
            # Extract the region of interest (ROI)
            roi = image[i : i + ker_h, j : j + ker_w]
            # Element-wise multiplication and sum
            output[i, j] = np.sum(roi * kernel)

    return output


def main():
    # --- Configuration ---
    image_file = "image_32x32.hex"
    weights_file = "kernels_3x3.hex"

    # 1. Load and reshape the 32x32 image
    print("Loading image data...")
    raw_image = load_hex_file(image_file, expected_size=32 * 32)
    image = raw_image.reshape((32, 32))
    print(f"Image loaded successfully. Shape: {image.shape}")

    # 2. Load the weights/kernels
    print("\nLoading kernel weights...")
    raw_weights = load_hex_file(weights_file, expected_size=3 * 3)

    # Calculate how many 3x3 kernels are in the file
    num_kernels = raw_weights.size // (3 * 3)
    kernels = raw_weights.reshape((num_kernels, 3, 3))
    print(f"Loaded {num_kernels} kernel(s) of size 3x3.")

    # 3. Perform Convolution for each kernel
    print("\nRunning convolutions...")
    for idx in range(num_kernels):
        kernel = kernels[idx]
        print(f"\n--- Processing Kernel {idx + 1}/{num_kernels} ---")
        print("Kernel Weights:")
        print(kernel)

        # Run convolution
        result = manual_convolution_2d(image, kernel)

        print(f"Output Feature Map Shape: {result.shape} (Valid padding)")
        print("Resulting Feature Map (Snippet of top-left 5x5):")
        print(result[:5, :5])

        # Optional: Save results to a file if needed
        # np.savetxt(f"output_kernel_{idx+1}.txt", result, fmt='%d')


if __name__ == "__main__":
    main()
