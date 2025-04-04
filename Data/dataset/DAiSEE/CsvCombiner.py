import pandas as pd


def combine_and_remove_duplicates(csv_path_1, csv_path_2, output_path):
    """
    Reads two CSVs, combines them, and removes duplicate rows.
    Writes the result to 'output_path'.
    """
    # 1) Read the two CSVs
    df1 = pd.read_csv(csv_path_1)
    df2 = pd.read_csv(csv_path_2)

    # 2) Concatenate
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # 3) Remove duplicate rows
    combined_df.drop_duplicates(inplace=True)

    # 4) Save to output
    combined_df.to_csv(output_path, index=False)
    print(f"Combined CSV saved to: {output_path}")


# Example usage:
if __name__ == "__main__":
    csv1 = r"C:\Users\bhbha\OneDrive - The Academic College of Tel-Aviv Jaffa - MTA\Desktop\DaiseeData\Labels\combined1.csv"
    csv2 = r"C:\Users\bhbha\OneDrive - The Academic College of Tel-Aviv Jaffa - MTA\Desktop\DaiseeData\Labels\ValidationLabels.csv"
    output = r"C:\Users\bhbha\OneDrive - The Academic College of Tel-Aviv Jaffa - MTA\Desktop\DaiseeData\Labels\Labels.csv"
    combine_and_remove_duplicates(csv1, csv2, output)

