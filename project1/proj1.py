import sys
print(sys.executable)
import pickle

# All of the imports that are going to be used throughout the model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("https://raw.githubusercontent.com/Mik-Nowak-05/Alzheimers_Disease_Prediction_Model/refs/heads/main/alzheimers_disease_data.csv")
df.head()

df.shape
df = df.drop(['PatientID', 'DoctorInCharge'], axis=1) # Dropping the Columns that have IDs or Data that is unpredictable
df.shape

df.head(10)

cat_df = df.select_dtypes(include=[object]) # Checking whether categorical columns exist inside the data frame
cat_df.describe

z_scores = np.abs(stats.zscore(df, nan_policy='omit')) # Calculating the z-scores

z_scores_df = pd.DataFrame(z_scores, columns=df.columns, index=df.index) # Converting back to the DataFrame so that we can use it later
z_mean_per_row = z_scores_df.mean(axis=1) # Mean z-score per row

z_mean_per_row.describe()

Q1 = z_scores_df.quantile(0.25) # Compute the lower Q1
Q3 = z_scores_df.quantile(0.75) # Computer the upper Q3
IQR = Q3 - Q1 # Calculate the IQR, as we will need that for outlier boundaries

lower_bound = Q1 - 1.5 * IQR # Defining lower bound for outlier
upper_bound = Q3 + 1.5 * IQR # Defining upper bound for outlier

outliers = (z_scores_df < lower_bound) | (z_scores_df > upper_bound) # Identifying outlier cells per column based on the above bounds

outlier_fraction = outliers.mean(axis=1) # Calculating fractions of the outlier values for each row

df_outliers = df[outlier_fraction >= 0.5] # Dropping rows that are >= 50% filled with outliers
df_outliers.shape # No outliers inside the dataset that would fill the threshold of 50%

corr = df.corr(numeric_only=True)['Diagnosis'].sort_values(ascending=False) # Computing the correlations with the target variable,
                                                                                            # i.e. Diagnosis for better insight
print(corr)

# Splitting data into train and test
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
 
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
 
y_train = df_train.Diagnosis.values
y_val = df_val.Diagnosis.values
y_test = df_test.Diagnosis.values
 
del df_train['Diagnosis']
del df_val['Diagnosis']
del df_test['Diagnosis'] 


# Scale features
scaler = StandardScaler()
df_train_scaled = scaler.fit_transform(df_train)
df_val_scaled = scaler.transform(df_val)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(df_train_scaled, y_train)

# Predict
# probability that Diagnosis = 1
y_pred = model.predict_proba(df_val_scaled)[:, 1]
alz_prediction = (y_pred >= 0.5)

# Accuracy
(y_val == alz_prediction).mean()

# Accuracy and Dummy model
len(y_val)
(y_val == alz_prediction).sum()
(y_val == alz_prediction).mean()

# Evaluate the model on different thresholds
thresholds = np.linspace(0, 1, 21)
thresholds

thresholds = np.linspace(0, 1, 21)
scores = []
 
for t in thresholds:
    score = accuracy_score(y_val, y_pred >= t)
    print('%.2f %.3f' % (t, score))
    scores.append(score)
# threshold of 0.55 gives accuracy of 0.849
plt.plot(thresholds,scores)

# people who will develop Alzheimer's
actual_positive = (y_val == 1)
# people who will not develop Alzheimer's
actual_negative = (y_val == 0)

t = 0.55
predict_positive = (y_pred >= t)
predict_negative = (y_pred < t)

predict_positive & actual_positive
 
tp = (predict_positive & actual_positive).sum()
tp

tn = (predict_negative & actual_negative).sum()
tn
 
fp = (predict_positive & actual_negative).sum()
fp

fn = (predict_negative & actual_positive).sum()
fn

auc = roc_auc_score(y_val, y_pred)
print("AUC-ROC:", auc)

models = {
    "LogisticRegression": (
        LogisticRegression(max_iter=500),
        {"C": [0.1, 1, 10]}
    ),

    "DecisionTree": (
        DecisionTreeClassifier(),
        {"max_depth": [3, 5, 10], "min_samples_leaf": [1, 5, 10]}
    ),

    "RandomForest": (
        RandomForestClassifier(),
        {"n_estimators": [100, 300],
         "max_depth": [None, 10, 20]}
    ),

    "GradientBoosting": (
        GradientBoostingClassifier(),
        {"n_estimators": [100, 200],
         "learning_rate": [0.05, 0.1]}
    ),
}

# Cross-validation evaluation helper
def evaluate_with_cv(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    return scores.mean(), scores.std()

# Hyperparameter tuning with GridSearchCV
summary_rows = []

for name, (model, param_grid) in models.items():

    print(f"🔍 Tuning {name}...")

    # Pipeline with scaling (tree models ignore the scaler)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    grid = GridSearchCV(pipe, param_grid={"model__" + k: v for k, v in param_grid.items()},
                        cv=5, scoring="roc_auc", n_jobs=-1)

    grid.fit(df_train, y_train)

    best_model = grid.best_estimator_
    mean_score, std_score = evaluate_with_cv(best_model, df_train, y_train)

    summary_rows.append({
        "Model": name,
        "Best Params": grid.best_params_,
        "CV Mean AUC": mean_score,
        "CV Std": std_score,
    })

# Summary results table 
df_results = pd.DataFrame(summary_rows)
print(df_results.sort_values("CV Mean AUC", ascending=False))

df_sorted = df_results.sort_values("CV Mean AUC", ascending=False)
print(df_sorted.iloc[0].Model)

# save the model to pickle
model_file = 'random_forest_tuned' 

f_out = open(model_file, 'wb') 
pickle.dump((df_sorted.iloc[0].Model), f_out)
model_file
f_out.close()

# load the model from pickle
with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)
 
model

# Trying out the Random Forest Classifier
random_forest = RandomForestClassifier(random_state=42)

random_forest.fit(df_train, y_train)

y_pred = random_forest.predict(df_test)

# Testing the Predictions
print("----------------------------------------------------------")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("----------------------------------------------------------")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n----------------------------------------------------------")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("----------------------------------------------------------")

# Realised that the scores are great, apart from the Recall which I would like to be higher
# Using RandomizedSearchCV to fine-tune the Random Forest Classifier
params = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 5, 10, 15, 20, 30],
    'min_samples_split': [2, 4, 6, 8, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'max_features': ['sqrt', 'log2', 0.5, None],
    'bootstrap': [True, False],
    'class_weight': [None, 'balanced', 'balanced_subsample']
}

random_forest_cv = RandomizedSearchCV(
    estimator=random_forest,
    param_distributions=params,
    n_iter=50,
    scoring='recall', # Optimising the Recall
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_forest_cv.fit(df_train, y_train)

# Checking the best parameters:
print("----------------------------------------------------------")
print("Best Recall Score (CV):", random_forest_cv.best_score_)
print("----------------------------------------------------------")
print("\nBest Parameters:\n", random_forest_cv.best_params_)
print("----------------------------------------------------------")

# Trying the Fine-tuned Random Forest Classifier
random_forest_tuned = RandomForestClassifier(
    n_estimators=200,
    min_samples_split=2,
    min_samples_leaf=3,
    max_features=None,
    max_depth=10,
    class_weight='balanced',
    bootstrap=True,
    random_state=42
)

random_forest_tuned.fit(df_train, y_train)

y_pred = random_forest_tuned.predict(df_test)

# Testing the Predictions
print("----------------------------------------------------------")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("----------------------------------------------------------")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Joblib for Usage in Production
joblib.dump(random_forest_tuned, "random_forest_model_tuned.pkl") # Saving the fine-tuned model for production (Overall Accuracy of ~96%)