import numpy as np
from scipy.signal import convolve2d, correlate2d


def convolution2d_filter(input_path, weights_path, output_path):
    # 1. Read 1-digit-per-line 4-bit hex weights
    with open(weights_path, "r") as f:
        weight_tokens = f.read().replace("{", "").replace("}", "").split()

    # Parse all hex digits and apply 4-bit sign extension
    all_raw_w = np.array([int(t, 16) & 0x0F for t in weight_tokens], dtype=np.int8)
    all_k = np.where(all_raw_w & 0x08, all_raw_w - 16, all_raw_w).astype(np.int8)

    # 2. Read and parse hex file into 32x32 image
    with open(input_path, "r") as f:
        hex_data = f.read().replace("{", "").replace("}", "").split()
    img = np.array([int(h, 16) for h in hex_data], dtype=np.uint8).reshape(32, 32)

    # List to hold all hex lines across all 16 kernels
    combined_hex_output = []

    # 3. Process each of the 16 kernels sequentially
    num_kernels = 16
    for k_idx in range(num_kernels):
        start_idx = k_idx * 9
        end_idx = start_idx + 9

        if end_idx > len(all_k):
            print(f"Warning: Found only {k_idx} kernels in weights file. Stopping.")
            break

        w = all_k[start_idx:end_idx].reshape(3, 3)

        # Built-in 2D Convolution with zero-padding
        accumulator = correlate2d(img, w, mode="same", boundary="fill", fillvalue=0)

        # 4. ReLU Activation
        output_img = np.where(accumulator < 0, 0, accumulator).astype(np.uint16)

        # 5. Format as uppercase HEX strings with 4 digits padding
        kernel_hex = [f"{val:04X}" for val in output_img.flatten()]
        combined_hex_output.extend(kernel_hex)

    # 6. Write everything out into a single master hex file
    with open(output_path, "w") as f:
        f.write("\n".join(combined_hex_output))

    print(
        f"Successfully processed {num_kernels} kernels into a single file: '{output_path}' ({len(combined_hex_output)} lines)."
    )


if __name__ == "__main__":
    convolution2d_filter(
        "weights/verification/image.hex",
        "weights/verification/weights.hex",
        "weights/verification/output_pixels_full.hex",
    )
