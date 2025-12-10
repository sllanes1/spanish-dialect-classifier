import os

def split_slang(input_file, output_folder, prefix):
    """Splits a slang file (one sentence per line) into one-file-per-sentence."""
    print(f"\nProcessing {input_file}...")

    # Read lines
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Found {len(lines)} sentences.")

    # Create output files
    for i, sentence in enumerate(lines, start=1):
        filename = os.path.join(output_folder, f"{prefix}_slang_{i}.txt")

        with open(filename, "w", encoding="utf-8") as out:
            out.write(sentence)

    print(f"Created {len(lines)} files in {output_folder}. Done!")


# -------------------------------------------------
# CONFIGURE YOUR FILE PATHS HERE
# -------------------------------------------------

# MX slang file
mx_input = "data/clean/mx/mx_slang.txt"
mx_output = "data/clean/mx"

# ES slang file
es_input = "data/clean/es/es_slang.txt"
es_output = "data/clean/es"

# -------------------------------------------------
# RUN SPLITTING FOR BOTH
# -------------------------------------------------

split_slang(mx_input, mx_output, "mx")
split_slang(es_input, es_output, "es")