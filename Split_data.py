def split_data(X_features_df, Y_vector, split_type=None, test_size=0.2, random_state=42, root_path='.', cv_folds=5):
    import numpy as np
    import pandas as pd
    import os
    from sklearn.model_selection import train_test_split, StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler

    df = X_features_df.copy()
    df['label'] = Y_vector
    df['global_index'] = np.arange(len(df))

    if 'child_ID' not in df.columns:
        raise ValueError("X_features_df must contain a 'child_ID' column.")

    non_feature_cols = ['filename', 'child_ID', 'label', 'global_index']
    feature_cols = [col for col in df.columns if col not in non_feature_cols]

    if split_type == 'within_group':
        train_idx, test_idx, buffer = [], [], []

        for child_id in df['child_ID'].unique():
            group_df = df[df['child_ID'] == child_id].reset_index(drop=True)
            label_counts = group_df['label'].value_counts()

            if len(label_counts) < 2 or min(label_counts) < 2:
                if len(group_df) >= 2:
                    idx_train, idx_test = train_test_split(
                        group_df.index,
                        test_size=test_size,
                        random_state=random_state
                    )
                else:
                    buffer.append(group_df)
                    continue
            else:
                try:
                    idx_train, idx_test = train_test_split(
                        group_df.index,
                        test_size=test_size,
                        stratify=group_df['label'],
                        random_state=random_state
                    )
                except ValueError:
                    buffer.append(group_df)
                    continue

            train_idx.extend(group_df.loc[idx_train, 'global_index'])
            test_idx.extend(group_df.loc[idx_test, 'global_index'])

        train_df = df.loc[train_idx].reset_index(drop=True)
        test_df = df.loc[test_idx].reset_index(drop=True)

        if buffer:
            buffer_df = pd.concat(buffer).reset_index(drop=True)
            target_label_ratio = df['label'].mean()

            for _, row in buffer_df.iterrows():
                label = row['label']
                train_ratio = (train_df['label'].sum() + (label == 1)) / (len(train_df) + 1)
                test_ratio = (test_df['label'].sum() + (label == 1)) / (len(test_df) + 1)
                train_dev = abs(train_ratio - target_label_ratio)
                test_dev = abs(test_ratio - target_label_ratio)

                if test_dev < train_dev:
                    test_df = pd.concat([test_df, pd.DataFrame([row])], ignore_index=True)
                else:
                    train_df = pd.concat([train_df, pd.DataFrame([row])], ignore_index=True)

        # ⚠ Normalize within each child separately
        print("[🔄] Normalizing each child separately (with NaN protection)...")

        def normalize_per_child(df_):
            dfs = []
            for cid, group in df_.groupby('child_ID'):
                group = group.copy()
                features = group[feature_cols]

                means = features.mean()
                stds = features.std()

                # Identify constant features (std = 0 or NaN)
                const_features = stds[(stds == 0) | (stds.isna())].index.tolist()
                norm_features = stds[(stds != 0) & (~stds.isna())].index.tolist()

                if const_features:
                    print(f"[child {cid}] ⚠️ Skipped normalization for {len(const_features)} constant features:")
                    print(" ", const_features)

                # Normalize only non-constant features
                group.loc[:, norm_features] = (features[norm_features] - means[norm_features]) / stds[norm_features]

                # Leave constant features untouched
                dfs.append(group)

            result = pd.concat(dfs).reset_index(drop=True)

            # Optional: check for NaNs anyway (just in case)
            n_before = result.shape[0]
            result = result.dropna(subset=feature_cols)
            n_after = result.shape[0]

            if n_before != n_after:
                print(f"⚠️ Dropped {n_before - n_after} rows with NaNs after normalization.")

            return result

        train_df = normalize_per_child(train_df)
        test_df = normalize_per_child(test_df)


    elif split_type == 'stratified_groupkfold':


        def stratified_group_train_test_split(df, test_size=0.2, random_state=42):
            grouped = df.groupby('child_ID')
            stats = grouped['label'].agg(['count', 'sum']).reset_index()
            stats['neg'] = stats['count'] - stats['sum']
            stats['pos'] = stats['sum']
            total = stats['count'].sum()
            total_pos = stats['pos'].sum()
            target_ratio = total_pos / total

            rng = np.random.RandomState(random_state)
            best_diff, best_groups = float('inf'), None

            for _ in range(1000):
                shuffled = stats.sample(frac=1, random_state=rng.randint(0, 10000))
                test_groups, test_count, test_pos = [], 0, 0

                for _, row in shuffled.iterrows():
                    if test_count >= total * test_size:
                        break
                    test_groups.append(row['child_ID'])
                    test_count += row['count']
                    test_pos += row['pos']

                test_ratio = test_pos / test_count if test_count else 0
                diff = abs(test_ratio - target_ratio)

                if diff < best_diff:
                    best_diff, best_groups = diff, test_groups
                    if diff <= 0.02:
                        break

            df_test = df[df['child_ID'].isin(best_groups)].reset_index(drop=True)
            df_train = df[~df['child_ID'].isin(best_groups)].reset_index(drop=True)
            return df_train, df_test

        train_df, test_df = stratified_group_train_test_split(df, test_size=test_size, random_state=random_state)

        # ✅ Normalize safely: fit on train, transform on test
        print("[🔄] Normalizing with StandardScaler (fit on train only)...")
        scaler = StandardScaler()
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    elif split_type == 'cv':
        sgkf = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        X = df.drop(columns=['label', 'global_index'])
        y = df['label']
        groups = df['child_ID']
        splits = []

        for i, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)

            print(f"[🔄] Normalizing CV fold {i + 1}...")
            scaler = StandardScaler()
            train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
            test_df[feature_cols] = scaler.transform(test_df[feature_cols])

            splits.append((train_df, test_df))

            train_df.to_csv(os.path.join(root_path, f'train_cv_fold_{i + 1}.csv'), index=False)
            test_df.to_csv(os.path.join(root_path, f'test_cv_fold_{i + 1}.csv'), index=False)
            print(f"✅ Saved fold {i + 1}")

        return splits

    else:
        raise ValueError("split_type must be one of: 'within_group', 'stratified_groupkfold', 'cv'")

    # Save and return
    print(f"\n=== Split Summary ({split_type}) ===")
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    print(f"Train labels: {train_df['label'].value_counts().to_dict()}")
    print(f"Test  labels: {test_df['label'].value_counts().to_dict()}")

    train_path = os.path.join(root_path, f'train_split_{split_type}.csv')
    test_path = os.path.join(root_path, f'test_split_{split_type}.csv')
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"💾 Saved to {train_path} and {test_path}")

    return (
        train_df.drop(columns=['label', 'global_index']),
        test_df.drop(columns=['label', 'global_index']),
        train_df['label'].values,
        test_df['label'].values,
        train_df['child_ID'].values,
        test_df['child_ID'].values,
        train_df.index.values,
        test_df.index.values
    )
