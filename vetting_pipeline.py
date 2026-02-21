import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
from skrebate import ReliefF

os.makedirs("plots", exist_ok=True)

def display_correlation_matrix(X, feature_names=None, plot=False, split_type="split"):
    corr_matrix = pd.DataFrame(X).corr(method='spearman').values
    if plot:
        plt.figure(figsize=(10, 8))
        plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar()
        if feature_names:
            plt.xticks(range(len(feature_names)), feature_names, rotation=90)
            plt.yticks(range(len(feature_names)), feature_names)
        plt.title("Feature Correlation Matrix (Spearman)")
        plt.tight_layout()
        plt.savefig(f"plots/correlation_matrix_{split_type}.png")
        plt.close()
    return corr_matrix

def plot_feature_ranking(relief_model, split_type, feature_names=None, top_k=20):
    scores = relief_model.feature_importances_
    top_k = min(top_k, len(scores))
    sorted_indices = np.argsort(scores)[::-1][:top_k]
    top_scores = scores[sorted_indices]
    top_names = [feature_names[i] if feature_names else f"F{i}" for i in sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(top_names[::-1], top_scores[::-1])
    plt.xlabel("ReliefF Score")
    plt.title(f"Top {top_k} Features by ReliefF ({split_type.replace('_', ' ').title()})")
    plt.tight_layout()
    plt.savefig(f"plots/feature_ranking_{split_type}.png")
    plt.close()

    return sorted_indices

def vet_features(X_train, Y_train, split_type, X_test=None, k=20, plot_corr=False, feature_names=None, root_path='.'):
    n_neighbors = min(20, max(5, len(Y_train) // 10))
    full_relief = ReliefF(n_neighbors=n_neighbors, n_features_to_select=X_train.shape[1])
    full_relief.fit(X_train, Y_train)
    relief_scores = full_relief.feature_importances_

    corr_matrix = display_correlation_matrix(X_train, feature_names=feature_names, plot=plot_corr, split_type=split_type)

    triu_indices = np.triu_indices(corr_matrix.shape[0], k=1)
    corr_values = corr_matrix[triu_indices]
    high_corr_mask = np.abs(corr_values) > 0.8
    high_corr_pairs = list(zip(triu_indices[0][high_corr_mask], triu_indices[1][high_corr_mask]))

    to_remove = set()
    for i, j in high_corr_pairs:
        if relief_scores[i] < relief_scores[j]:
            to_remove.add(i)
        else:
            to_remove.add(j)

    print(f"\n⚠️ Removing {len(to_remove)} highly correlated features (|r| > 0.8) based on ReliefF importance.")
    if feature_names:
        for i, j in high_corr_pairs:
            if i in to_remove:
                print(f"  - Dropping {feature_names[i]} (r={corr_matrix[i,j]:.2f}) over {feature_names[j]})")
            elif j in to_remove:
                print(f"  - Dropping {feature_names[j]} (r={corr_matrix[i,j]:.2f}) over {feature_names[i]})")

    keep_indices = [i for i in range(X_train.shape[1]) if i not in to_remove]
    X_train_reduced = X_train[:, keep_indices]
    X_test_reduced = X_test[:, keep_indices] if X_test is not None else None
    reduced_feature_names = [feature_names[i] for i in keep_indices] if feature_names else None

    top_k = min(k, X_train_reduced.shape[1])
    relief = ReliefF(n_neighbors=n_neighbors, n_features_to_select=top_k)
    relief.fit(X_train_reduced, Y_train)
    X_train_vetted = relief.transform(X_train_reduced)

    top_k_indices = np.argsort(relief.feature_importances_)[::-1][:top_k]
    X_test_filtered = X_test_reduced[:, top_k_indices] if X_test_reduced is not None else None
    selected_feature_names = [reduced_feature_names[i] for i in top_k_indices] if reduced_feature_names else None

    if selected_feature_names:
        feature_scores = relief.feature_importances_[top_k_indices]
        vetted_df = pd.DataFrame({
            'Feature': selected_feature_names,
            'ReliefF_Score': feature_scores
        })
        vetted_path = os.path.join(root_path, f"vetted_features_{split_type}.csv")
        vetted_df.to_csv(vetted_path, index=False)
        print(f"✅ Saved vetted feature list to: {vetted_path}")

    plot_feature_ranking(relief, split_type, feature_names=reduced_feature_names, top_k=top_k)

    return X_train_vetted, X_test_filtered, selected_feature_names

def _vet_fold(train_df, test_df, split_type, k, plot_corr, root_path):
    group_col = 'child_ID'
    label_col = 'label'
    exclude_cols = [group_col, 'filename', label_col]

    # 🔍 Debugging step — check for bad column values
    print("Checking types of feature columns...")
    for col in train_df.columns:
        if col not in exclude_cols:
            bad_rows = train_df[~train_df[col].apply(
                lambda x: isinstance(x, (int, float, np.integer, np.floating)) or pd.isna(x)
            )]
            if not bad_rows.empty:
                print(f"⚠️ Column '{col}' contains non-numeric or sequence data. Sample:")
                print(bad_rows[[col]].head())

    def is_valid_numeric(col):
        return is_numeric_dtype(train_df[col]) and all(
            isinstance(x, (int, float, np.integer, np.floating)) or pd.isna(x)
            for x in train_df[col]
        )

    feature_names = [
        col for col in train_df.columns
        if col not in exclude_cols and is_valid_numeric(col)
    ]

    X_train = train_df[feature_names].values.astype(np.float32)
    Y_train = train_df[label_col].values
    X_test = test_df[feature_names].values.astype(np.float32)
    Y_test = test_df[label_col].values

    X_train_vetted, X_test_vetted, selected_feature_names = vet_features(
        X_train, Y_train,
        split_type=split_type,
        X_test=X_test,
        k=k,
        plot_corr=plot_corr,
        feature_names=feature_names,
        root_path=root_path
    )

    train_vetted_path = os.path.join(root_path, f"X_train_vetted_{split_type}.csv")
    test_vetted_path = os.path.join(root_path, f"X_test_vetted_{split_type}.csv")

    pd.DataFrame(X_train_vetted, columns=selected_feature_names).to_csv(train_vetted_path, index=False)
    pd.DataFrame(X_test_vetted, columns=selected_feature_names).to_csv(test_vetted_path, index=False)

    print(f"✅ Saved X_train_vetted to: {train_vetted_path}")
    print(f"✅ Saved X_test_vetted to: {test_vetted_path}")

    return X_train_vetted, Y_train, X_test_vetted, Y_test, selected_feature_names

def vetting_pipeline(split_type: str, root_path='.', k=20, plot_corr=True):
    if split_type == 'cv':
        fold_idx = 1
        while True:
            train_path = os.path.join(root_path, f"train_cv_fold_{fold_idx}.csv")
            test_path = os.path.join(root_path, f"test_cv_fold_{fold_idx}.csv")
            if not os.path.exists(train_path) or not os.path.exists(test_path):
                if fold_idx == 1:
                    raise FileNotFoundError(f"No CV fold files found. Expected {train_path}")
                break
            print(f"\n📂 Processing CV Fold {fold_idx}")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            _vet_fold(train_df, test_df, split_type=f"cv_fold{fold_idx}", k=k, plot_corr=plot_corr, root_path=root_path)
            fold_idx += 1
        return

    train_path = os.path.join(root_path, f"train_split_{split_type}.csv")
    test_path = os.path.join(root_path, f"test_split_{split_type}.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Could not find split files: {train_path}, {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return _vet_fold(train_df, test_df, split_type, k, plot_corr, root_path)
