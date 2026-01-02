import os
import shutil
import argparse
import pandas as pd


def split_images_by_cdr(input_dir_path, output_dir_path, excel_file_path, cdr="cdrtot"):
    assert cdr in ["cdrtot", "cdrsum"], "cdr must be 'cdrtot' or 'cdrsum'"


    df = pd.read_excel(excel_file_path)

    filename_col = "filename"
    cdr_col = "cdrtot" if cdr == "CDRTOT" else "cdrsum"

    if filename_col not in df.columns:
        raise ValueError(f"'{filename_col}' column not found in Excel")

    if cdr_col not in df.columns:
        raise ValueError(f"'{cdr_col}' column not found in Excel")

    filename_to_cdr = dict(
        zip(df[filename_col].astype(str), df[cdr_col])
    )

    out_0 = os.path.join(output_dir_path, "0")
    out_1 = os.path.join(output_dir_path, "1")

    os.makedirs(out_0, exist_ok=True)
    os.makedirs(out_1, exist_ok=True)

    # Traverse images
    copied_0, copied_1, skipped = 0, 0, 0

    for fname in os.listdir(input_dir_path):
        if not fname.lower().endswith(".png"):
            continue

        if fname not in filename_to_cdr:
            skipped += 1
            continue

        cdr_value = filename_to_cdr[fname]

        if pd.isna(cdr_value):
            skipped += 1
            continue

        src_path = os.path.join(input_dir_path, fname)

        if cdr_value == 0:
            shutil.copy2(src_path, os.path.join(out_0, fname))
            copied_0 += 1

        elif cdr_value > 0:
            shutil.copy2(src_path, os.path.join(out_1, fname))
            copied_1 += 1

        else:
            skipped += 1

    print("\nFinished")
    print(f"Copied to class 0 (no dementia): {copied_0}")
    print(f"Copied to class 1 (dementia):    {copied_1}")
    print(f"Skipped:                        {skipped}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--excel_file",
        type=str,
        required=True
    )

    parser.add_argument(
        "--cdr",
        type=str,
        choices=["cdrtot", "cdrsum"],
        default="cdrtot"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    split_images_by_cdr(
        input_dir_path=args.input_dir,
        output_dir_path=args.output_dir,
        excel_file_path=args.excel_file,
        cdr=args.cdr
    )


if __name__ == "__main__":
    main()
