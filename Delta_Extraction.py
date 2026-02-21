import os
import pandas as pd

def extract_deltas(root_path, balanced_csv_path):
    # Load the balanced CSV
    pairs_df = pd.read_csv(balanced_csv_path)

    # Build normalized delta filenames
    def build_expected_filename(pre, post):
        return f"delta_{os.path.splitext(pre)[0]}_vs_{os.path.splitext(post)[0]}.csv".strip().lower()

    pairs_df['expected_delta_filename'] = pairs_df.apply(
        lambda row: build_expected_filename(row['pre_filename'], row['post_filename']),
        axis=1
    )

    # Create lookup from delta filename to binary label (already 0 or 1)
    label_lookup = pairs_df.set_index('expected_delta_filename')['label'].to_dict()

    # ==========================
    # Delta File Regeneration
    # ==========================
    print("Regenerating delta files...")

    generating_path = r"C:\Users\97254\OneDrive - mail.tau.ac.il\שולחן העבודה\לימודים\שנה ד\פרוייקט גמר\OpenFace\OpenFace_2.2.0_win_x86\CSV_processed_sorted_new"

    for idx, row in pairs_df.iterrows():
        session_id = str(row['session-id'])
        pre_file_path = os.path.join(generating_path, session_id, row['pre_filename'])
        post_file_path = os.path.join(generating_path, session_id, row['post_filename'])

        session_folder = os.path.join(root_path, session_id)
        os.makedirs(session_folder, exist_ok=True)

        delta_file_name = row['expected_delta_filename']
        delta_file_path = os.path.join(session_folder, delta_file_name)

        missing_files = []
        if not os.path.isfile(pre_file_path):
            missing_files.append(f"PRE file not found: {pre_file_path}")
        if not os.path.isfile(post_file_path):
            missing_files.append(f"POST file not found: {post_file_path}")

        if missing_files:
            print(f"[Session {session_id}] Skipping delta generation due to missing files:")
            for msg in missing_files:
                print(f"  - {msg}")
            continue

        try:
            pre_df = pd.read_csv(pre_file_path)
            post_df = pd.read_csv(post_file_path)

            if pre_df.shape != post_df.shape:
                print(f"Shape mismatch in session {session_id}, skipping.")
                continue

            delta_df = post_df - pre_df
            delta_df.to_csv(delta_file_path, index=False)
            print(f"Saved delta file: {delta_file_path}")

        except Exception as e:
            print(f"Error processing delta for session {session_id}: {e}")

    # ==========================
    # Feature Extraction
    # ==========================
    print("\nCollecting regenerated delta files...")

    def list_all_delta_files(root_dir):
        return [os.path.join(subdir, file)
                for subdir, _, files in os.walk(root_dir)
                for file in files if file.startswith('delta_') and file.endswith('.csv')]

    actual_delta_files = list_all_delta_files(root_path)

    all_features = []

    for file_path in actual_delta_files:
        file_name = os.path.basename(file_path).strip().lower()
        label = label_lookup.get(file_name, None)

        if label is None:
            print(f"Warning: No label found for {file_name}, skipping.")
            continue

        try:
            df = pd.read_csv(file_path)
            if df.shape[0] != 1:
                print(f"Warning: Expected 1 row in {file_name}, found {df.shape[0]}, using first row only.")
            feature_row = df.iloc[0].copy()

            child_id = os.path.basename(os.path.dirname(file_path))
            feature_row['filename'] = file_name
            feature_row['child_ID'] = child_id
            feature_row['label'] = label

            all_features.append(feature_row)
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

    if not all_features:
        print("No valid features extracted. Aborting saving.")
        return

    # Save features and labels
    x_features_df = pd.DataFrame(all_features)
    y_vector_df = x_features_df[['filename', 'label']].copy()
    x_features_df = x_features_df.drop(columns=['label'])

    # Ensure column order
    cols = ['child_ID', 'filename'] + [col for col in x_features_df.columns if col not in ['child_ID', 'filename']]
    x_features_df = x_features_df[cols]

    # Save to disk
    x_features_df.to_csv(os.path.join(root_path, 'x_features.csv'), index=False)
    y_vector_df.to_csv(os.path.join(root_path, 'y_vector.csv'), index=False)

    print(f"\n✅ Saved {len(x_features_df)} feature rows and labels to:\n  - x_features.csv\n  - y_vector.csv")
