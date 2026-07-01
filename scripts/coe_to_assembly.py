import os
import re


def coe_to_assembly(coe_file_path, output_s_path, symbol_name="cnn_data"):
    """
    Converts a 32-bit hexadecimal .coe file into a RISC-V compatible .S assembly file.
    """
    if not os.path.exists(coe_file_path):
        print(f"Error: File {coe_file_path} not found.")
        return

    words = []

    # Process the COE file line by line
    with open(coe_file_path, "r") as f:
        for line in f:
            line = line.strip()
            # Ignore headers or empty lines
            if not line or "memory_initialization" in line:
                continue

            # Clean up the hex string (remove commas, semicolons, and comments if any)
            cleaned = re.sub(r"[^0-9a-fA-F]", "", line)

            if cleaned:
                # Format to a standard 0x hex representation
                words.append(f"    .word 0x{cleaned.lower()}")

    # Write out to the assembly (.S) file
    with open(output_s_path, "w") as f:
        f.write(f".global {symbol_name}\n")
        f.write(".section .rodata\n")
        f.write(".align 2\n\n")
        f.write(f"{symbol_name}:\n")
        f.write("\n".join(words))
        f.write("\n")  # Final newline

    print(f"Successfully generated {output_s_path} with {len(words)} 32-bit entries.")


# Example usage:
# This converts your uploaded 'image.coe' into 'cnn_image.S'
coe_to_assembly(
    "weights/export/output_pixels.coe",
    "weights/export/cnn_output.S",
    symbol_name="cnn_input_image",
)
