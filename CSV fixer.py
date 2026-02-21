import os
import pandas as pd

def match_files_to_subfolders(data_path, CSV_path, save_updated_csv=False):
    # Load the CSV
    df = pd.read_csv(CSV_path)

    # Ensure the 'file_names' column exists
    if 'file_name' not in df.columns:
        raise ValueError("'file_name' column not found in the CSV.")

    # Build a mapping from file name (without extension) to subfolder name
    file_to_subfolder = {}
    for root, dirs, files in os.walk(data_path):
        subfolder = os.path.basename(root)
        for file in files:
            name_without_ext = os.path.splitext(file)[0]
            file_to_subfolder[name_without_ext] = subfolder

    # Match and add the 'subfolder' column
    df['subfolder'] = df['file_name'].map(file_to_subfolder)

    # Optionally save the updated CSV
    if save_updated_csv:
        df.to_csv(CSV_path, index=False)

    return df

data_path = r"C:\Users\97254\OneDrive - mail.tau.ac.il\שולחן העבודה\לימודים\שנה ד\פרוייקט גמר\The_Child_Affective_Facial_Expression_(CAFE)_set\sessions"
CSV_path = r"C:\Users\97254\OneDrive - mail.tau.ac.il\שולחן העבודה\לימודים\שנה ד\פרוייקט גמר\The_Child_Affective_Facial_Expression_(CAFE)_set\The Child Affective Facial Expression (CAFE) set.csv"
updated_df = match_files_to_subfolders(data_path, CSV_path, save_updated_csv=True)
print(updated_df.head())
