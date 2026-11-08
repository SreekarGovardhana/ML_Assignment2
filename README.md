# Student Outcome Predictor

Predicting whether a university student will drop out, stay enrolled or graduate, using five ML models and a Streamlit app.

- **GitHub repository:** https://github.com/SreekarGovardhana/ML_Assignment2
- **Live app:** https://mlassignment2-g5ubhdpuaqptt9b3z35y52.streamlit.app

## a. Problem statement

Universities often find out that a student is struggling only after the student has already dropped out. This dataset describes each student at enrolment and at the end of their first two semesters with 36 numeric attributes: background and family details, how they were admitted, whether fees are paid, how many curricular units they enrolled in and passed, their grades, and the economic conditions at the time. The goal is to predict each student's final outcome (**Dropout**, **Enrolled** or **Graduate**), so that students at risk can be identified early and offered support. I trained five classifiers (Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes and SVM) on the same train/test split and compared them on six metrics.

## b. Dataset description

| Item | Value |
|---|---|
| Source | UCI Machine Learning Repository, Predict Students' Dropout and Academic Success (ID 697), Polytechnic Institute of Portalegre, Portugal |
| Instances | 4,424 (no missing values) |
| Features | 36 numeric, for example Admission grade, Age at enrollment, Tuition fees up to date, Scholarship holder, Curricular units 1st sem (approved), Curricular units 2nd sem (grade), Unemployment rate, GDP. Categorical attributes (course, nationality, parents' occupation and so on) are already integer-coded in the source file |
| Target | `Target`, one of 3 classes: Dropout, Enrolled, Graduate |
| Class counts | Graduate 2,209 · Dropout 1,421 · Enrolled 794 |

The data was split 80/20 with stratification (`random_state=42`), which gives 3,539 training rows and 885 test rows. Every model is a pipeline with `StandardScaler` followed by the classifier, so the scaler is fitted on the training rows only and saved inside each model file. The target labels are encoded with a `LabelEncoder`, which is also saved. The 885 test rows are saved as `test_data.csv`.

Precision, Recall and F1 are **weighted** averages (each class counts in proportion to its size). AUC is one-vs-rest weighted, computed from the predicted probabilities. MCC is the multi-class version. Because the averages are weighted, Recall is always equal to Accuracy, so MCC is the main metric I used to compare models: it accounts for all three classes and for the imbalance.

## c. GitHub Repository Link

https://github.com/SreekarGovardhana/ML_Assignment2

Files in the repository:

- `app.py`: the Streamlit app, which loads the saved models and never retrains
- `ML_Assignment2.ipynb`: EDA, preprocessing, tuning, training, evaluation, and saving the models and the test split
- `requirements.txt`: dependencies, with scikit-learn pinned to the version used for training
- `test_data.csv`: the 885 held-out rows (36 features + `Target`)
- `data.csv`: the full UCI dataset
- `model/`: the five saved `.joblib` pipelines, plus `label_encoder.joblib`, `feature_columns.json`, `best_params.json` and `metrics_comparison.csv`

To run it locally: `pip install -r requirements.txt`, run the notebook top to bottom (it recreates `model/` and `test_data.csv`), then `streamlit run app.py`.

## d. Models used

Results on the 885 test rows:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7684 | 0.8961 | 0.7500 | 0.7684 | 0.7531 | 0.6150 |
| Decision Tree | 0.7299 | 0.8669 | 0.7145 | 0.7299 | 0.7176 | 0.5506 |
| KNN | 0.6678 | 0.8082 | 0.6483 | 0.6678 | 0.6553 | 0.4465 |
| Naive Bayes | 0.6588 | 0.8118 | 0.6334 | 0.6588 | 0.6417 | 0.4279 |
| Support Vector Machine | 0.7548 | 0.8901 | 0.7476 | 0.7548 | 0.7464 | 0.5946 |

Mean rank across the six metrics (1 = best): Logistic Regression 1.00, SVM 2.00, Decision Tree 3.00, KNN 4.17, Naive Bayes 4.83.

Settings: each model was tuned with `GridSearchCV` (5-fold stratified CV on the training set, scored by macro F1). The grids were: Logistic Regression C ∈ {0.01, 0.1, 1, 10}; Decision Tree max_depth ∈ {4, 6, 8, 10, None}, min_samples_leaf ∈ {1, 5, 10, 20}, criterion ∈ {gini, entropy}; KNN k ∈ {5, 9, 15, 21, 31}, weights ∈ {uniform, distance}; Gaussian Naive Bayes var_smoothing ∈ {1e-9, 1e-7, 1e-5, 1e-3}; SVM C ∈ {0.5, 1, 5}, kernel ∈ {rbf, linear}. The values chosen by the search (saved in `model/best_params.json`) were: Logistic Regression C=1; Decision Tree criterion=gini, max_depth=6, min_samples_leaf=5; KNN k=5, uniform weights; Gaussian Naive Bayes var_smoothing=1e-9; SVM linear kernel, C=5.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | The best model on every metric: accuracy 0.7684, AUC 0.8961, precision 0.7500, F1 0.7531 and MCC 0.6150. The gap between training and test accuracy was very small, so it generalises well. Many of the strongest features (curricular units approved and grades in each semester, tuition fees up to date) push the outcome in one direction, which suits a linear model. Its weakest class was **Enrolled**. |
| Decision Tree | Third overall, with accuracy 0.7299, AUC 0.8669 and MCC 0.5506. It beat KNN and Naive Bayes on every metric but stayed below Logistic Regression and SVM. Its splits rely heavily on one feature: Curricular units 2nd sem (approved) accounts for about two-thirds of the tree's total feature importance, with Tuition fees up to date a distant second. Training accuracy was clearly higher than test accuracy, which shows some overfitting even after tuning the depth and leaf size. Its AUC is also held back because a tree gives coarse, leaf-level probabilities. |
| KNN | Fourth overall: accuracy 0.6678, AUC 0.8082, F1 0.6553 and MCC 0.4465. It is ahead of Naive Bayes on accuracy, precision, recall, F1 and MCC, but has the lowest AUC of the five. Scaling helps, but many features are integer codes for categories (course, nationality, parents' occupation), so the distance between two students on those features does not mean much. |
| Naive Bayes | The weakest model: lowest accuracy (0.6588), precision (0.6334), F1 (0.6417) and MCC (0.4279), although its AUC (0.8118) is slightly above KNN's. Gaussian Naive Bayes assumes the features are independent and normally distributed. Neither holds here: the 1st- and 2nd-semester features are strongly correlated with each other, and many features are discrete codes. |
| Support Vector Machine | A close second: accuracy 0.7548, AUC 0.8901, F1 0.7464 and MCC 0.5946, all within about 0.02 of Logistic Regression. The grid search picked a **linear** kernel over RBF, which shows the classes are separated fairly linearly once the features are scaled. That explains why it lands so close to Logistic Regression: both draw linear boundaries, and the SVM's margin-based fit gave no extra benefit here. It was also the slowest model to train. |
| Overall winner | **Logistic Regression.** It has the best Accuracy, AUC, Precision, Recall, F1 and MCC, with SVM a close second. KNN and Naive Bayes were the weakest. For every model, **Enrolled** was the hardest class to predict, because those students sit between dropouts and graduates and it is also the smallest class (18% of the data). |

## e. Streamlit app

- **Sidebar:** upload a CSV (or tick "Use bundled test_data.csv"), download the sample test file, and pick a model from the dropdown.
- **Model evaluation tab:** the six metrics for the chosen model, a confusion matrix, a per-class classification report and a per-class recall chart.
- **Compare all models tab:** a table with every metric for all five models on the uploaded file, a bar chart of any one metric, and a download button for the table.
- **Predictions tab:** the actual and predicted outcome for each row with the class probabilities, an option to show only misclassified rows, and a CSV download.
- **Uploaded data tab:** row and column counts, a preview of the rows and the class balance.

The uploaded CSV needs the 36 feature columns and a `Target` column with the values Dropout, Enrolled or Graduate. Comma- and semicolon-separated files both work, and extra columns or a different column order are fine. If a required column is missing or `Target` contains an unknown label, the app shows an error. If the file does not contain all three classes, AUC is shown as n/a. The app only loads the saved models from `model/`. It never trains.