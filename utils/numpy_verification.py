import numpy as np
from scipy.signal import convolve2d


def convolution2d_filter(input_path, weights_path, output_path):
    # 1. Read 1-digit-per-line 4-bit hex weights
    with open(weights_path, "r") as f:
        weight_tokens = f.read().replace("{", "").replace("}", "").split()

    # Parse hex digits and apply 4-bit sign extension (subtract 16 if >= 8)
    raw_w = np.array([int(t, 16) & 0x0F for t in weight_tokens[:9]], dtype=np.int8)
    k = np.where(raw_w & 0x08, raw_w - 16, raw_w).astype(np.int8)

    # Flip weights into spatial arrangement matching your C implementation
    w = k.reshape(3, 3)

    # 2. Read and parse hex file into 32x32 image
    with open(input_path, "r") as f:
        hex_data = f.read().replace("{", "").replace("}", "").split()
    img = np.array([int(h, 16) for h in hex_data], dtype=np.uint8).reshape(32, 32)

    # 3. Built-in 2D Convolution with zero-padding
    accumulator = convolve2d(img, w, mode="same", boundary="fill", fillvalue=0)

    # 4. ReLU Activation
    output_img = np.where(accumulator < 0, 0, accumulator).astype(np.uint16)

    # 5. Format and write directly as uppercase HEX strings with 4 digits padding
    hex_output = [f"{val:04X}" for val in output_img.flatten()]
    with open(output_path, "w") as f:
        f.write("\n".join(hex_output))


if __name__ == "__main__":
    convolution2d_filter(
        "weights/verification/image.hex",
        "weights/verification/weights.hex",
        "weights/verification/output_pixels.hex",
    )
