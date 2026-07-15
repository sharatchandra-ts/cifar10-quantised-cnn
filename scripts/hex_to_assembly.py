import os
import re


def hex_to_assembly(
    input_hex_path,
    output_s_path,
    symbol_name="cnn_data",
    word_width_bits=32,
    pixel_width_bits=8,
):
    """
    Converts a continuous hex file directly into a RISC-V assembly (.S) file.

    The first byte in the input becomes the least-significant byte of each word.
    """

    chars_per_word = word_width_bits // 4
    chars_per_pixel = pixel_width_bits // 4

    if word_width_bits % pixel_width_bits != 0:
        raise ValueError("Pixel width must divide word width.")

    if not os.path.exists(input_hex_path):
        print(f"Error: File {input_hex_path} not found.")
        return

    # Read only hex digits
    with open(input_hex_path, "r") as f:
        hex_digits = re.sub(r"[^0-9a-fA-F]", "", f.read())

    words = []

    for i in range(0, len(hex_digits), chars_per_word):
        word = hex_digits[i : i + chars_per_word]

        # Pad the last word if needed
        if len(word) < chars_per_word:
            word = word.zfill(chars_per_word)

        pixel_list = [
            word[j : j + chars_per_pixel] for j in range(0, len(word), chars_per_pixel)
        ]

        word = "".join(reversed(pixel_list))

        words.append(f"    .word 0x{word.lower()}")

    with open(output_s_path, "w") as f:
        f.write(f".global {symbol_name}\n")
        f.write(".section .rodata\n")
        f.write(".align 2\n\n")
        f.write(f"{symbol_name}:\n")
        f.write("\n".join(words))
        f.write("\n")

    print(f"Successfully generated {output_s_path}")
    print(f"Wrote {len(words)} {word_width_bits}-bit words.")


if __name__ == "__main__":
    hex_to_assembly(
        "weights/verification/weights.hex",
        "weights/export/asm/layer1_weights.S",
        word_width_bits=32,
        pixel_width_bits=4,
    )
