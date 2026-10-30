# Multiclass Classification of Diabetes Using Machine Learning

Classifying patients as non-diabetic, prediabetic or diabetic from 11 routine clinical measurements, and comparing four standard classifiers on a public dataset of 264 patients. Course project for CS 403/603 Machine Learning at IIT Indore (2025).

## Summary

| | |
|---|---|
| Data | 264 patients (96 non-diabetic, 40 prediabetic, 128 diabetic), 11 features |
| Evaluation | Stratified 70/30 train/test split (184/80 patients, `random_state=42`); 5-fold cross-validation on the training set |
| Best model | Random forest (100 trees, default settings) |
| Test accuracy | 0.95 (76 of 80 test patients) |
| Prediabetic recall | 0.92 (11 of 12 test patients) |
| Top features (random-forest importance) | HbA1c 0.417, BMI 0.270, age 0.114 |

## Data

`diabetes_raw.csv`: 264 rows, 11 features and a `Class` label (0 = non-diabetic, 1 = prediabetic, 2 = diabetic).

Features: `Gender`, `AGE`, `Urea`, `Cr` (creatinine), `HbA1c`, `Chol` (total cholesterol), `TG` (triglycerides), `HDL`, `LDL`, `VLDL`, `BMI`. There are no missing values.

Source: the Multiclass Diabetes Dataset, downloaded from [Kaggle](https://www.kaggle.com/datasets/yasserhessein/multiclass-diabetes-dataset). The same 264-patient set is published on Mendeley Data:

- Sahid, A., Babar, M. U. H. and Uddin, M. P. (2024). *Multiclass Diabetes Dataset*. Mendeley Data, V1. [doi:10.17632/jpp8bsjgrm.1](https://doi.org/10.17632/jpp8bsjgrm.1). CC BY 4.0. A refined subset of:
- Rashid, A. (2020). *Diabetes Dataset*. Mendeley Data, V1. [doi:10.17632/wj9rwkp9c2.1](https://doi.org/10.17632/wj9rwkp9c2.1). CC BY 4.0. Patients of the Medical City Hospital laboratory and the Specialized Center for Endocrinology and Diabetes, Al-Kindy Teaching Hospital, Iraq.

## Method

All steps are in `Diabetes_Prediction_model.ipynb`, saved with its outputs.

1. **Exploration:** class balance, correlation heatmap, feature distributions by class.
2. **Pre-processing:** stratified 70/30 split; features standardised with `StandardScaler` fitted on the training set only.
3. **Models:** logistic regression (multinomial), random forest (100 trees), gradient boosting (100 estimators) and an RBF-kernel SVM, all with scikit-learn default hyperparameters (no tuning).
4. **Evaluation:** accuracy and support-weighted precision, recall and F1 on the held-out test set; per-class reports; confusion matrices; one-vs-rest ROC curves for the random forest; 5-fold cross-validation accuracy on the training set as a stability check.

## Results

Held-out test set of 80 patients (29 non-diabetic, 12 prediabetic, 39 diabetic):

| Model | Accuracy | Precision | Recall | F1 | 5-fold CV accuracy (training set) |
|---|---|---|---|---|---|
| Random forest | 0.9500 | 0.9509 | 0.9500 | 0.9498 | 0.9836 ± 0.0134 |
| Gradient boosting | 0.9375 | 0.9378 | 0.9375 | 0.9373 | 0.9730 ± 0.0171 |
| Logistic regression | 0.9000 | 0.8990 | 0.9000 | 0.8977 | 0.8913 ± 0.0296 |
| SVM (RBF) | 0.8750 | 0.8767 | 0.8750 | 0.8734 | 0.8315 ± 0.0464 |

Precision, recall and F1 are weighted by class support; ± is the standard deviation across the five folds.

**Random forest errors:** all 29 non-diabetic patients were classified correctly; 11 of 12 prediabetic patients (one predicted diabetic); 36 of 39 diabetic patients (two predicted non-diabetic, one prediabetic). One-vs-rest ROC AUC: 0.999 (non-diabetic), 0.999 (prediabetic), 0.983 (diabetic).

**Feature importance** (random forest, impurity-based): HbA1c 0.417, BMI 0.270, age 0.114, creatinine 0.036, VLDL 0.033, cholesterol 0.030, triglycerides 0.030, urea 0.028, LDL 0.022, HDL 0.015, gender 0.005.

## Limitations

- **Small test set.** With 80 test patients, one error moves accuracy by 1.25 points; the prediabetic class has only 12 test patients. Random forest and gradient boosting differ by a single patient (76 vs 75 correct).
- **One split, one seed.** The results come from a single train/test split, with no confidence intervals and no external validation set.
- **One source population.** All patients come from one data source in Iraq.
- **Data quality.** A few values look physiologically implausible (for example HbA1c below 3% and a total cholesterol of 0) and were not cleaned.
- **Importance is not causation.** Impurity-based importances describe this model on this dataset only.

## Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook Diabetes_Prediction_model.ipynb
```

The first cell reads the CSV from GitHub. To use the copy in this repository, set `url = "diabetes_raw.csv"`.

## Files

| File | Contents |
|---|---|
| `Diabetes_Prediction_model.ipynb` | Full analysis, with outputs |
| `diabetes_raw.csv` | Dataset (264 patients) |

## Maintainer

Avinash Bharti, PhD scholar, Computer Science and Engineering, IIT Indore · [LinkedIn](https://www.linkedin.com/in/avinashkamalbharti/) · avinashkamalbharti@gmail.com

We thank Dr. Puneet Gupta for his guidance.
