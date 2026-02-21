import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    accuracy_score
)
from sklearn.preprocessing import label_binarize
from sklearn.utils import compute_sample_weight
from sklearn.metrics import roc_curve


def compute_metrics(cm):
    """
    Compute accuracy and sensitivity metrics from confusion matrix.

    Args:
        cm: Confusion matrix

    Returns:
        accuracy class 0, sensitivity class 0, accuracy class 1, sensitivity class 1
    """
    acc_0 = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    sens_0 = cm[0, 0] / (cm[0, 0] + cm[1, 0]) if (cm[0, 0] + cm[1, 0]) > 0 else 0
    acc_1 = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    sens_1 = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0

    return acc_0, sens_0, acc_1, sens_1


def train_model(X_selected, Y_train, model_type='boosting', save_path=None, verbose=True):
    if verbose:
        print(f"Training model: {model_type}...")

    if model_type == 'svm':
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        model = GridSearchCV(
            SVC(probability=True, class_weight='balanced'),
            param_grid, cv=5, verbose=verbose, n_jobs=-1
        )
        sample_weight = None

    elif model_type == 'rf':
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 5, 10]
        }
        base_rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        model = GridSearchCV(base_rf, param_grid, cv=5, verbose=verbose, n_jobs=-1)
        sample_weight = None

    elif model_type == 'boosting':
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [2, 3, 5]
        }
        base_boost = GradientBoostingClassifier(random_state=42)
        model = GridSearchCV(base_boost, param_grid, cv=5, verbose=verbose, n_jobs=-1)
        sample_weight = compute_sample_weight(class_weight='balanced', y=Y_train)

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # ✅ Fit the model (conditionally pass sample_weight if available)
    if sample_weight is not None:
        model.fit(X_selected, Y_train, sample_weight=sample_weight)
    else:
        model.fit(X_selected, Y_train)

    if verbose:
        print("✅ Model training complete.")
        if hasattr(model, 'best_params_'):
            print(f"Best parameters: {model.best_params_}")

    if save_path:
        joblib.dump(model, save_path)
        if verbose:
            print(f"💾 Model saved to {save_path}")

    return model

def evaluate_model(model, X_test, Y_test, class_names=None, model_name=None, root_path="."):
    plots_dir = os.path.join(root_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    full_classes = np.array([0, 1])
    probs = model.predict_proba(X_test)

    if probs.shape[1] == 1:
        if np.all(Y_test == 0):
            probs = np.hstack([probs, 1 - probs])
        else:
            probs = np.hstack([1 - probs, probs])

    roc_aucs = []
    optimal_thresholds = {}

    plt.figure(figsize=(10, 6))
    for i, c in enumerate(full_classes):
        y_true = (Y_test == c).astype(int)
        fpr, tpr, thresholds = roc_curve(y_true, probs[:, i])
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)

        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        optimal_thresholds[c] = thresholds[best_idx]

        plt.plot(fpr, tpr, label=f"Class {c} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"ROC Curve - {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"ROC_{model_name}.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    for i, c in enumerate(full_classes):
        y_true = (Y_test == c).astype(int)
        precision, recall, _ = precision_recall_curve(y_true, probs[:, i])
        ap = average_precision_score(y_true, probs[:, i])
        plt.plot(recall, precision, label=f"Class {c} (AP={ap:.2f})")

    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"PR_Curve_{model_name}.png"))
    plt.close()

    preds = np.argmax(probs, axis=1)
    cm = confusion_matrix(Y_test, preds, labels=full_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names or full_classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"Confusion_Matrix_{model_name}.png"))
    plt.close()

    tp = cm[1, 1]
    fn = cm[1].sum() - tp
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    return roc_aucs, sensitivity, cm, optimal_thresholds

def train_and_evaluate_all(root_path, k=20):
    model_types = ['boosting', 'svm', 'rf']
    summary_records = []
    plot_dir = os.path.join(root_path, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # === Standard splits ===
    split_types = ['within_group', 'stratified_groupkfold']
    for split in split_types:
        print(f"\n==================== SPLIT: {split} ====================")
        x_train_path = os.path.join(root_path, f"X_train_vetted_{split}.csv")
        x_test_path = os.path.join(root_path, f"X_test_vetted_{split}.csv")
        y_train_path = os.path.join(root_path, f"train_split_{split}.csv")
        y_test_path = os.path.join(root_path, f"test_split_{split}.csv")

        if not all(os.path.exists(p) for p in [x_train_path, x_test_path, y_train_path, y_test_path]):
            print(f"❌ Missing vetted files for split: {split}. Please run vetting first.")
            continue

        X_train = pd.read_csv(x_train_path).values
        X_test = pd.read_csv(x_test_path).values
        y_train = pd.read_csv(y_train_path)['label'].values
        y_test = pd.read_csv(y_test_path)['label'].values

        for model_type in model_types:
            print(f"\n🔧 Training {model_type.upper()} model...")
            model = train_model(X_selected=X_train, Y_train=y_train, model_type=model_type, verbose=False)

            aucs, sensitivity_1, cm, thresholds = evaluate_model(
                model, X_test, y_test,
                class_names=["0", "1"],
                model_name=f"{model_type}_{split}",
                root_path=root_path
            )

            acc_0, sens_0, acc_1, sens_1 = compute_metrics(cm)

            summary_records.append({
                "Split": split,
                "Model": model_type,
                "AUC_Class0": round(aucs[0], 3),
                "AUC_Class1": round(aucs[1], 3),
                "Acc_0": acc_0, "Sens_0": sens_0,
                "Acc_1": acc_1, "Sens_1": sens_1
            })

    # === CV folds ===
    print("\n==================== SPLIT: CV ====================")
    fold_idx = 1
    cm_totals = {model_type: np.zeros((2, 2), dtype=int) for model_type in model_types}
    mean_fpr = np.linspace(0, 1, 100)
    roc_tprs = {model_type: [] for model_type in model_types}
    roc_aucs_all = {model_type: [] for model_type in model_types}

    while True:
        train_path = os.path.join(root_path, f"train_cv_fold_{fold_idx}.csv")
        test_path = os.path.join(root_path, f"test_cv_fold_{fold_idx}.csv")
        x_train_path = os.path.join(root_path, f"X_train_vetted_cv_fold{fold_idx}.csv")
        x_test_path = os.path.join(root_path, f"X_test_vetted_cv_fold{fold_idx}.csv")

        if not all(os.path.exists(p) for p in [train_path, test_path, x_train_path, x_test_path]):
            break

        X_train = pd.read_csv(x_train_path).values
        X_test = pd.read_csv(x_test_path).values
        y_train = pd.read_csv(train_path)['label'].values
        y_test = pd.read_csv(test_path)['label'].values

        for model_type in model_types:
            print(f"\n🔁 CV Fold {fold_idx}: Training {model_type.upper()} model...")
            model = train_model(X_selected=X_train, Y_train=y_train, model_type=model_type, verbose=False)
            probs = model.predict_proba(X_test)
            preds = model.predict(X_test)
            aucs, sensitivity_1, cm, thresholds = evaluate_model(
                model, X_test, y_test,
                class_names=["0", "1"],
                model_name=f"{model_type}_cv_fold{fold_idx}",
                root_path=root_path
            )
            precision_per_class = precision_score(y_test, preds, average=None, zero_division=0)
            recall_per_class = recall_score(y_test, preds, average=None, zero_division=0)
            f1_per_class = f1_score(y_test, preds, average=None, zero_division=0)
            acc_0, sens_0, acc_1, sens_1 = compute_metrics(cm)

            summary_records.append({
                "Split": f"cv_fold{fold_idx}",
                "Model": model_type,
                "AUC_Class0": round(aucs[0], 3),
                "AUC_Class1": round(aucs[1], 3),
                "Acc_0": acc_0,
                "Sens_0": sens_0,
                "Acc_1": acc_1,
                "Sens_1": sens_1,
                "Precision_0": precision_per_class[0],
                "Precision_1": precision_per_class[1],
                "Recall_0": recall_per_class[0],
                "Recall_1": recall_per_class[1],
                "F1_0": f1_per_class[0],
                "F1_1": f1_per_class[1],
            })

            cm_totals[model_type] += cm

            # Average ROC collection
            fpr, tpr, _ = roc_curve(y_test, probs[:, 1])
            tpr_interp = np.interp(mean_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            roc_tprs[model_type].append(tpr_interp)
            roc_aucs_all[model_type].append(auc(fpr, tpr))

        fold_idx += 1

    # Save summary
    summary_df = pd.DataFrame(summary_records)
    summary_path = os.path.join(root_path, "model_evaluation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n📊 Saved model evaluation summary to: {summary_path}")

    # === Aggregated Stats for CV Folds ===
    print("\n📈 Aggregated Cross-Fold Statistics:")

    def is_cv_fold(x):
        return x.startswith("cv_fold")

    cv_df = summary_df[summary_df['Split'].apply(is_cv_fold)]
    if not cv_df.empty:
        agg_df = (
            cv_df.groupby("Model")
            .agg({
                "AUC_Class0": ['mean', 'std'],
                "AUC_Class1": ['mean', 'std'],
                "Acc_0": ['mean', 'std'],
                "Sens_0": ['mean', 'std'],
                "Acc_1": ['mean', 'std'],
                "Sens_1": ['mean', 'std'],
                "Precision_0": ['mean', 'std'],
                "Precision_1": ['mean', 'std'],
                "Recall_0": ['mean', 'std'],
                "Recall_1": ['mean', 'std'],
                "F1_0": ['mean', 'std'],
                "F1_1": ['mean', 'std'],
            })
        )

        agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
        agg_df.reset_index(inplace=True)

        extended_summary_path = os.path.join(root_path, "cv_fold_aggregated_metrics_extended.csv")
        agg_df.to_csv(extended_summary_path, index=False)

        print(f"✅ Extended CV summary saved to: {extended_summary_path}")
        print(agg_df.to_string(index=False))

        # === Save averaged confusion matrices ===
        for model_type, cm_total in cm_totals.items():
            n_folds = fold_idx - 1
            avg_cm = cm_total / n_folds
            disp = ConfusionMatrixDisplay(confusion_matrix=avg_cm.astype(int), display_labels=["0", "1"])
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f"Mean Confusion Matrix Across CV - {model_type.upper()}")
            plt.tight_layout()
            path = os.path.join(plot_dir, f"mean_confusion_matrix_cv_{model_type}.png")
            plt.savefig(path)
            print(f"📊 Saved average confusion matrix to: {path}")
            plt.close()

        # === Save averaged ROC curves ===
        plt.figure(figsize=(10, 6))
        for model_type in model_types:
            tprs = np.array(roc_tprs[model_type])
            if tprs.size == 0:
                continue
            mean_tpr = tprs.mean(axis=0)
            std_tpr = tprs.std(axis=0)
            mean_auc = np.mean(roc_aucs_all[model_type])
            std_auc = np.std(roc_aucs_all[model_type])

            plt.plot(mean_fpr, mean_tpr, label=f"{model_type.upper()} (AUC={mean_auc:.2f}±{std_auc:.2f})")
            plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.2)

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.title("Mean ROC Curve Across CV Folds")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        avg_plot_path = os.path.join(plot_dir, "mean_roc_across_cv.png")
        plt.savefig(avg_plot_path)
        print(f"📈 Saved averaged ROC curve to: {avg_plot_path}")
        plt.close()

    else:
        print("⚠️ No cross-validation folds found for aggregation.")
        print("\n📊 Generating averaged CV metrics plots...")
