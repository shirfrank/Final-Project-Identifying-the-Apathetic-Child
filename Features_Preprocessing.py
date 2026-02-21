def preprocess_features(root_path, remove_outliers=False):
    import pandas as pd
    import numpy as np
    import os
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import zscore

    x_features_path = os.path.join(root_path, 'x_features.csv')
    y_vector_path = os.path.join(root_path, 'y_vector.csv')

    x_df = pd.read_csv(x_features_path)
    y_df = pd.read_csv(y_vector_path)

    print(f"Loaded {x_df.shape[0]} X samples and {y_df.shape[0]} Y labels.\n")

    # Filename alignment check
    x_filenames = set(x_df['filename'])
    y_filenames = set(y_df['filename'])
    aligned_filenames = x_filenames & y_filenames

    x_df = x_df[x_df['filename'].isin(aligned_filenames)].reset_index(drop=True)
    y_df = y_df[y_df['filename'].isin(aligned_filenames)].reset_index(drop=True)

    print(f"After alignment: {x_df.shape[0]} samples remain.\n")

    # Drop filename column for feature processing
    feature_data = x_df.drop(columns=['filename']).copy()

    # Convert all numeric columns to float64
    numeric_columns = feature_data.select_dtypes(include=[np.number]).columns
    feature_data[numeric_columns] = feature_data[numeric_columns].astype(np.float64)

    # Report initial NaNs and Infs
    n_nans_before = feature_data.isna().sum().sum()
    n_infs_before = feature_data[numeric_columns].apply(np.isinf).sum().sum()
    print(f"Initial NaNs: {n_nans_before}")
    print(f"Initial Infs: {n_infs_before}\n")

    if n_nans_before > 0 or n_infs_before > 0:
        print("Starting cleaning process...")

    # Replace Infs with NaNs for uniform treatment
    feature_data[numeric_columns] = feature_data[numeric_columns].replace([np.inf, -np.inf], np.nan)

    # Fill NaNs with column mean
    feature_data[numeric_columns] = feature_data[numeric_columns].fillna(feature_data[numeric_columns].mean())

    # Report remaining NaNs and Infs after cleaning
    n_nans_after = feature_data.isna().sum().sum()
    n_infs_after = feature_data[numeric_columns].apply(np.isinf).sum().sum()
    print(f"Remaining NaNs after cleaning: {n_nans_after}")
    print(f"Remaining Infs after cleaning: {n_infs_after}\n")

    if n_nans_after == 0 and n_infs_after == 0:
        print("Cleaning successful. No remaining NaNs or Infs.")
    else:
        print("Warning: Some NaNs or Infs remain after cleaning.")

    # Print feature statistics before outlier removal/standardization
    print("\nFeature statistics after cleaning:")
    print(feature_data[numeric_columns].describe())

    # Optional: remove outliers using z-score
    if remove_outliers:
        z_scores = np.abs(zscore(feature_data[numeric_columns]))
        mask = (z_scores < 3).all(axis=1)
        original_shape = feature_data.shape[0]
        feature_data = feature_data[mask].reset_index(drop=True)
        y_df = y_df[mask].reset_index(drop=True)
        print(f"\nRemoved {original_shape - feature_data.shape[0]} outlier rows based on Z-score.")

    # Prepend filename back using concat (fixes fragmentation warning)
    feature_data = pd.concat([x_df[['filename']], feature_data], axis=1)

    # Save cleaned data
    clean_x_path = os.path.join(root_path, 'x_features_cleaned.csv')
    clean_y_path = os.path.join(root_path, 'y_vector_cleaned.csv')

    feature_data.to_csv(clean_x_path, index=False)
    y_df.to_csv(clean_y_path, index=False)

    print(f"\nFinal cleaned feature shape: {feature_data.shape}")
    print(f"Saved cleaned X features to: {clean_x_path}")
    print(f"Saved aligned Y vector to: {clean_y_path}")
