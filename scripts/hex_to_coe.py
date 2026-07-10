import os
import re


def hex_to_coe(
    input_hex_path,
    output_coe_path,
    word_width_bits=32,
    pixel_width_bits=8,
):
    """
    Converts a continuous hex file into a Xilinx .coe file based on specified bit width.

    :param input_hex_path: Path to the input file containing raw hex characters.
    :param output_coe_path: Path where the output .coe file will be written.
    :param word_width_bits: Target width of the BRAM in bits (e.g., 32, 64, 16).
    """
    chars_per_word = word_width_bits // 4
    chars_per_pixel = pixel_width_bits // 4

    if word_width_bits % pixel_width_bits != 0:
        raise ValueError("Pixel width must divide word width.")

    if not os.path.exists(input_hex_path):
        print(f"Error: Source file {input_hex_path} not found.")
        return

    # 1. Read and clean the input file to extract ONLY raw hex digits
    with open(input_hex_path, "r") as f:
        content = f.read()
        # Keep only valid hex characters (0-9, a-f, A-F)
        hex_digits = re.sub(r"[^0-9a-fA-F]", "", content)

    # 2. Chunk the continuous hex string into blocks matching the width
    coe_words = []

    for i in range(0, len(hex_digits), chars_per_word):
        word = hex_digits[i : i + chars_per_word]

        # Pad the final incomplete word
        if len(word) < chars_per_word:
            word = word.zfill(chars_per_word)

        pixel_list = [
            word[j : j + chars_per_pixel] for j in range(0, len(word), chars_per_pixel)
        ]

        word = "".join(reversed(pixel_list))

        coe_words.append(word.lower())

    if not coe_words:
        print("Warning: No valid hex digits found in the input file.")
        return

    # 3. Write out the Xilinx COE file
    with open(output_coe_path, "w") as f:
        f.write("memory_initialization_radix = 16;\n")
        f.write("memory_initialization_vector =\n")

        # Print all lines except the last one with a trailing comma
        for word in coe_words[:-1]:
            f.write(f"{word},\n")

        # The very last entry in a COE file must end with a semicolon
        f.write(f"{coe_words[-1]};\n")

    print(f"Successfully generated {output_coe_path}")
    print(
        f"Processed {len(hex_digits)} hex digits into {len(coe_words)} rows ({word_width_bits}-bit wide)."
    )


# ==========================================
# Example Usage:
# ==========================================
# Assuming you have a file 'raw_dump.hex' containing your stream of hex values:
# hex_to_coe("raw_dump.hex", "output_image.coe", word_width_bits=32)

if __name__ == "__main__":
    hex_to_coe(
        "weights/verification/output_pixels_full.hex",
        "weights/export/coe/output_pixels_full.coe",
        word_width_bits=32,
    )
