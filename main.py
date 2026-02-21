from Delta_Extraction import extract_deltas
from Features_Preprocessing import preprocess_features
from Split_data import split_data
from vetting_pipeline import vetting_pipeline
from Train_Evaluate import train_and_evaluate_all
import os
import pandas as pd

ROOT_PATH = r"C:\Users\97254\OneDrive - mail.tau.ac.il\שולחן העבודה\לימודים\שנה ד\פרוייקט גמר\delta_new"
BALANCED_CSV_PATH = r"C:\Users\97254\OneDrive - mail.tau.ac.il\שולחן העבודה\לימודים\שנה ד\פרוייקט גמר\The_Child_Affective_Facial_Expression_(CAFE)_set\emotion_image_pairs.csv"

def run_data_split(cv_folds=5):
    x_path = os.path.join(ROOT_PATH, "x_features_cleaned.csv")
    y_path = os.path.join(ROOT_PATH, "y_vector_cleaned.csv")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print("❌ Cleaned feature or label files not found. Please run preprocessing first.")
        return None

    X_df = pd.read_csv(x_path)
    Y_vec = pd.read_csv(y_path)['label'].values

    for split_type in ['within_group', 'stratified_groupkfold']:
        print(f"\nRunning split: {split_type}")
        split_data(
            X_features_df=X_df,
            Y_vector=Y_vec,
            split_type=split_type,
            test_size=0.2,
            random_state=42,
            root_path=ROOT_PATH
        )

    # Cross-validation splits
    print("\nRunning cross-validation split...")
    cv_splits = split_data(
        X_features_df=X_df,
        Y_vector=Y_vec,
        split_type='cv',
        test_size=0.2,  # not used in 'cv' mode
        random_state=42,
        root_path=ROOT_PATH,
        cv_folds=cv_folds
    )

    for i, (train_df, test_df) in enumerate(cv_splits):
        print(f"[CV Fold {i+1}] Train: {len(train_df)} | Test: {len(test_df)}")

    return cv_splits

def run_vetting(cv_folds=5):
    for split_type in ['within_group', 'stratified_groupkfold', 'cv']:
        print(f"\nRunning feature vetting for: {split_type}")
        vetting_pipeline(split_type, root_path=ROOT_PATH, k=20, plot_corr=True)

def main():
    print("\nChoose which pipeline step to run:")
    print("1. Delta Extraction")
    print("2. Preprocessing")
    print("3. Data Split (both within_group and stratified_groupkfold)")
    print("4. Feature Vetting (after split)")
    print("5. Train & Evaluate Models")
    print("6. Full Pipeline (Delta + Preprocess + Split + Vetting + Train)")

    choice = input("Enter your choice (1/2/3/4/5/6): ").strip()

    if choice == "1":
        print("Running Delta Extraction...")
        extract_deltas(ROOT_PATH, BALANCED_CSV_PATH)

    elif choice == "2":
        print("Running Preprocessing...")
        preprocess_features(ROOT_PATH)

    elif choice == "3":
        print("Running Data Splits...")
        run_data_split()

    elif choice == "4":
        print("Running Feature Vetting...")
        run_vetting()

    elif choice == "5":
        print("Running Cross-Validated Model Training and Evaluation...")
        train_and_evaluate_all(root_path=ROOT_PATH, k=20)

    elif choice == "6":
        print("Running Full Pipeline with Cross-Validation...")
        extract_deltas(ROOT_PATH, BALANCED_CSV_PATH)
        preprocess_features(ROOT_PATH)
        run_data_split()
        run_vetting()
        train_and_evaluate_all(root_path=ROOT_PATH, k=20)


    else:
        print("Invalid input. Please choose 1 to 6.")

if __name__ == "__main__":
    main()
