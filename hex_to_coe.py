def convert_rows_to_coe(hex_input_path, coe_output_path):
    try:
        # 1. Read all rows from your hex file
        with open(hex_input_path, "r") as f:
            # Clean up whitespace and ignore empty lines
            rows = [line.strip() for line in f if line.strip()]

        if not rows:
            print("Error: The source hex file is empty.")
            return

        # 2. Write out the formatted COE file
        with open(coe_output_path, "w") as f:
            # Specify the radix (base 16 for hexadecimal)
            f.write("memory_initialization_radix = 16;\n")
            f.write("memory_initialization_vector =\n")

            # Write all lines except the last one, separated by a comma and newline
            f.write(",\n".join(rows[:-1]))

            # Add a comma before the final element if there is more than one row
            if len(rows) > 1:
                f.write(",\n")

            # The very last line MUST end with a semicolon
            f.write(f"{rows[-1]};\n")

        print(f"Success! COE file generated at: '{coe_output_path}'")
        print(f"Total memory depth: {len(rows)} lines (Width: 256-bit)")

    except FileNotFoundError:
        print(f"Error: Could not find '{hex_input_path}'. Please check the path.")


# Convert your file
convert_rows_to_coe("image_rows.hex", "weights/coe/image_32x32.coe")
