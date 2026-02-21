# 🎓 Identifying the Apathetic Child: Facial Emotion Change Detection
### Final Project in Biomedical Engineering – Tel Aviv University
**Authors:** Shir Frank & Ofri Yacoby  
**Supervisor:** Prof. Mickey Scheinowitz  
**Clinical Partners:** Schneider Children’s Medical Center

---

## 📌 Project Overview
Apathy is defined as a lack of emotion, interest, or concern that is not due to low consciousness, emotional distress, or cognitive impairment. In pediatric settings, it is a critical but subtle indicator of serious illness, often missed in community healthcare.

This project develops an **automated machine learning tool** to support clinical assessment by identifying non-responsiveness to external visual stimulation through facial analysis.

---

## 🧠 Research Objectives
* **Data Collection:** Gathering clinical data of children’s reactions to visual stimuli (e.g., response-inducing videos) in an ER setting.
* **Apathy Modeling:** Utilizing the **Child Affective Facial Expression (CAFE)** dataset to build a classifier that distinguishes between "Reactive" and "Apathetic" transitions.
* **Feature Validation:** Testing if facial feature deltas ($\Delta$) can reliably indicate emotional change.

---

## 📂 Project Structure & Pipeline

| Module | Description |
| :--- | :--- |
| **`Couple_Generator.py`** | Creates photo pairs (Neutral vs. Emotional for "Reactive"; Neutral vs. Neutral for "Apathetic"). |
| **`Delta_Extraction.py`** | Extracts features via **OpenFace 2.0** and calculates the difference ($\Delta$) between post-stimulus and baseline features. |
| **`vetting_pipeline.py`** | Reduces 714 raw features down to the top 20 using **ReliefF** and Spearman correlation ($|\rho| > 0.8$). |
| **`Train_Evaluate.py`** | Implements and evaluates **Gradient Boosting**, **Random Forest**, and **SVM** models using cross-validation. |
| **`main.py`** | The central entry point for running the end-to-end automated pipeline. |

---

## 📊 Dataset & Analysis Logic
* **Dataset:** CAFE Dataset featuring children aged 2–8 years.
* **The Logic:** Instead of static classification, the model analyzes the **change** between a baseline "pre" photo and a post-stimulus "post" photo.
* **Features:** Focuses on **Facial Action Units (AUs)** such as Cheek Raisers (AU6), Nose Wrinklers (AU9), and Lip Corner Pullers (AU12).



---

## 📈 Performance Results
Evaluated across 1674 image pairs (approx. 1:8 apathetic to reactive ratio):

| Metric | Gradient Boosting | Random Forest | SVM |
| :--- | :--- | :--- | :--- |
| **AUC** | **0.9546 ± 0.014** | 0.954 ± 0.015 | 0.957 ± 0.008 |
| **Sensitivity (Apathetic)** | 0.680 ± 0.048 | **0.814 ± 0.095** | 0.660 ± 0.035 |
| **F1-Score (Apathetic)** | **0.717 ± 0.025** | 0.696 ± 0.085 | 0.679 ± 0.017 |



**Conclusion:** **Gradient Boosting** proved the most consistent classifier for identifying minimal emotional change between neutral expressions.

---

## 🧪 Clinical Implementation & Challenges
During clinical trials at Schneider Children’s Medical Center, several challenges were identified:
* **Low Incidence:** Rare occurrence of truly apathetic children in the ER during recruitment hours.
* **Ethical Constraints:** Ensuring medical care is not delayed for data collection.
* **Environmental Factors:** Parental distress and limited physician availability.

---

## 🛠 How to Run
1.  **Generate Pairs:** Run `python "Couple Generator.py"` to create the label vector from CAFE metadata.
2.  **Preprocessing:** Run `python Features_Preprocessing.py` to clean and align OpenFace outputs.
3.  **Run Pipeline:** Use `python main.py` and select **Option 6** for the full automated training and evaluation suite.

---
*This research serves as a proof-of-concept for the automated identification of pediatric apathy in community clinics.*
