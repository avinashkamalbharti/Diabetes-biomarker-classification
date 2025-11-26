# Multi-Class Classification of Diabetes using Clinical Biomarkers

## 📌 Project Overview
Diabetes is a rapidly growing global health challenge, where traditional diagnosis often misses the crucial **prediabetes** stage. This project investigates the use of Machine Learning (ML) techniques to automate the classification of patients into **Non-Diabetic, Pre-Diabetic, and Diabetic** categories based on standard clinical biomarkers.

The primary objective is to develop a diagnostic aid that interprets complex interactions between biochemical features (such as HbA1c, Urea, and Creatinine) to ensure earlier intervention.

## 🔬 Key Findings
* **Best Model:** Random Forest Classifier.
* **Performance:** Achieved an overall accuracy of **95.00%**.
* **Feature Importance:** The study identified **HbA1c** as the most critical biomarker for classification, followed by Cholesterol and Urea. Gender and HDL were found to have minimal impact on the prediction.

## 📂 Repository Structure
* `ML_Final0_2-2.ipynb`: The core Jupyter Notebook containing data preprocessing, EDA (Exploratory Data Analysis), model training, and evaluation.
* `diabetes_raw.csv`: The dataset containing 12 clinical features including HbA1c, BMI, Triglycerides (TG), and LDL/HDL levels.
* `ml.pptx`: Project presentation slides summarizing the problem motivation and final results.

## ⚙️ Methodology
The project follows a standard Data Science lifecycle:
1.  **Data Preprocessing:** Handling missing values, outlier detection, and feature scaling.
2.  **Exploratory Data Analysis:** Correlation analysis to identify relationships between biomarkers like BMI and blood sugar levels.
3.  **Model Training:** Comparative analysis of multiple classifiers including:
    * Random Forest (Best Performer)
    * Support Vector Machines (SVM)
    * Logistic Regression
4.  **Evaluation:** Models were assessed using Accuracy, Precision, Recall, and Confusion Matrices.

## 🚀 How to Run
1.  Clone this repository:
    ```bash
    git clone [https://github.com/avinashkamalbharti/Machine-Learning-Project](https://github.com/avinashkamalbharti/Machine-Learning-Project)
    ```
2.  Install the required dependencies:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
3.  Open the notebook:
    ```bash
    jupyter notebook ML_Final0_2-2.ipynb
    ```

## 🛠 Tech Stack
* **Language:** Python 3.x
* **Libraries:** Scikit-Learn (Modeling), Pandas (Data Manipulation), Matplotlib/Seaborn (Visualization).

---
*Developed as part of the PhD Coursework (CS 403/603) at IIT Indore.*

