import os
import pandas as pd

# Configuration
root_path = r"C:\Users\97254\OneDrive - mail.tau.ac.il\שולחן העבודה\לימודים\שנה ד\פרוייקט גמר\OpenFace\OpenFace_2.2.0_win_x86\delta"
balanced_csv_path = r"C:\Users\97254\OneDrive - mail.tau.ac.il\שולחן העבודה\לימודים\שנה ד\פרוייקט גמר\Balanced_Happy_Variant_Updated_Pre-Post_Pairs.csv"

# Load the balanced CSV
pairs_df = pd.read_csv(balanced_csv_path)

# Generate expected delta file names (without session folder)
def build_expected_filename(pre, post):
    pre_base = os.path.splitext(pre)[0]
    post_base = os.path.splitext(post)[0]
    return f'delta_{pre_base}_vs_{post_base}.csv'

pairs_df['expected_delta_filename'] = pairs_df.apply(
    lambda row: build_expected_filename(row['pre_filename'], row['post_filename']),
    axis=1
)

# Show some expected filenames
print("Sample expected delta filenames:")
print(pairs_df['expected_delta_filename'].head())
print(f"Total rows in CSV (not necessarily unique filenames): {len(pairs_df)}\n")

# Detect duplicates
duplicate_counts = pairs_df['expected_delta_filename'].value_counts()
duplicates = duplicate_counts[duplicate_counts > 1]
print("\nDuplicate expected delta filenames (referenced multiple times):")
print(duplicates)
print(f"Total duplicated filenames: {len(duplicates)}\n")

# Save duplicates to CSV
duplicates_df = duplicates.reset_index()
duplicates_df.columns = ['duplicate_filename', 'count']
duplicates_report_path = os.path.join(root_path, 'duplicate_expected_filenames.csv')
duplicates_df.to_csv(duplicates_report_path, index=False)
print(f"Duplicate files report saved to: {duplicates_report_path}\n")

# Function to list all actual delta files in all subdirectories
def list_all_delta_files(root_dir):
    delta_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.startswith('delta_') and file.endswith('.csv'):
                delta_files.append(file)
    return delta_files

# List actual delta files
actual_delta_files = set(list_all_delta_files(root_path))

# Show actual found delta files
print("Sample found delta files:")
print(list(actual_delta_files)[:5])
print(f"Total found delta files: {len(actual_delta_files)}\n")

# Compare unique expected files to actual files
expected_unique_files = set(pairs_df['expected_delta_filename'])
missing_files = expected_unique_files - actual_delta_files

# Report missing files
print("Missing unique delta files:")
for file in missing_files:
    print(file)

print(f"\nExpected unique delta files: {len(expected_unique_files)}")
print(f"Found unique delta files: {len(actual_delta_files)}")
print(f"Missing unique delta files: {len(missing_files)}")

# Save missing list to CSV
missing_df = pd.DataFrame({'missing_delta_filename': list(missing_files)})
missing_report_path = os.path.join(root_path, 'missing_delta_files.csv')
missing_df.to_csv(missing_report_path, index=False)
print(f"\nMissing files report saved to: {missing_report_path}")
