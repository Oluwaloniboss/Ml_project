# import all necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, RandomizedSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.preprocessing import QuantileTransformer

# load the dataset
dataset_2 = pd.read_csv("C:/Users/Dell/OneDrive/programming and coding/4-python/project_school_2/Company_House_Info.csv")

# print the dataset
print(dataset_2)

# check for other informations on the dataset

# check the shape of the dataset
print(f"Dataset Shape: {dataset_2.shape}\n")  # the dataset has a shape of 6665 rows by 96 columns

# check the information on the dataset
print(dataset_2.info())     # the dataset contains columns that are nominal which are either decimal or integer

# check if any of the columns can be converted to a category by looking at their unique values
print(dataset_2.nunique())     # all the dataset has more than 3000 unique values except for the target variable and net income flag column

# check for a summary of the dataset
print(dataset_2.describe().T)

# check for missing values
print(dataset_2.isnull().sum())  # all the columns have no missing values

# plot the missing values
plt.figure(figsize=(12, 6))
sns.heatmap(dataset_2.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# check the class distribution of the target variable
print("Class Distribution (Bankrupt?):")
print(dataset_2["Bankrupt?"].value_counts(normalize = True).apply(lambda x: f"{x:.2%}"))   # the class distribution is 96.71% non_bankrupt and 3.29% bankrupt which shows a major class imbalance
print(dataset_2["Bankrupt?"].value_counts())

# Plot class imbalance
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=dataset_2, x='Bankrupt?')

# Get total count
total = len(dataset_2)

# Add labels showing percentage for each class
for p in ax.patches:
    percentage = f"{100 * p.get_height() / total:.2f}%"  # Calculate percentage # type: ignore
    ax.annotate(percentage,
                (p.get_x() + p.get_width() / 2, p.get_height()), # type: ignore
                ha='center', va='bottom', fontsize=12, color='black', fontweight='bold')

plt.title("Class Imbalance Plot with Percentages")
plt.xlabel("Bankrupt?")
plt.ylabel("Count")
plt.show()

# Get value counts
class_counts = dataset_2["Bankrupt?"].value_counts()

# Create pie chart with counts and percentages
ax = class_counts.plot.pie(autopct = lambda p: f"{p:.1f}%\n({int(round(p * sum(class_counts) / 100))})", startangle = 90, figsize=(8, 8), colors=['lightcoral', 'lightskyblue'],
                            title = "Class Distribution (Percentage and Count)") # type: ignore

# Equal aspect ratio ensures pie is drawn as a circle
ax.axis('equal')

# Add legend
ax.legend(labels = [f"{label} ({count})" for label, count in zip(class_counts.index, class_counts)],
          title = "Classes",
          loc = "upper right")

plt.show()

# define parameter grid for GridSearchCV or RandomizedSearchCV
param_grid_xgb = {
    "n_estimators" : [100, 200, 300],
    "learning_rate" : [0.01, 0.1, 0.3],
    "max_depth" : [3, 6, 9],
    "subsample" : [0.7, 0.8, 1.0],
    "colsample_bytree" : [0.7, 0.8, 1.0],
            }

param_grid_rf = {
    "n_estimators" : [100, 200, 300],
    "min_samples_split" : [2, 5, 0.1],
    "max_depth" : [3, 4, 5, 6],
            }

kf = KFold(n_splits = 5, shuffle = False)

# first lets bulid a model and check the performance based on the raw data as it is
# select the features variable
X_raw = dataset_2.drop(["Bankrupt?"], axis = 1).values

# select the target variable
y_raw = dataset_2["Bankrupt?"].values

# split the data into 70 percent train and 30 percent test, using the features and the target variable of the raw data
X_raw_train, X_raw_test, y_raw_train, y_raw_test = train_test_split(X_raw, y_raw, test_size = 0.3, random_state = 21, stratify = y_raw) # type: ignore

# next we work with the XGBClassifier model on the raw data and see the accuracy
# create an instance of the XGBClassifier model
xgb_model_raw = XGBClassifier(scale_pos_weight = 30, random_state = 21)

# create the GridSearchCV for xgb model
xgb_model_raw_grid = RandomizedSearchCV(xgb_model_raw, param_grid_xgb, cv = kf, n_jobs = -1) # type: ignore

# fit the xgb model with the raw train data
xgb_model_raw_grid.fit(X_raw_train, y_raw_train)

# predict the X_raw_test with the xgb model
y_raw_pred_xgb = xgb_model_raw_grid.predict(X_raw_test)

# print the classification report of the XGBClassifier model on the raw dataset
xgb_raw_classification_report = classification_report(y_raw_test, y_raw_pred_xgb)
print(xgb_raw_classification_report)

# print the accuracy score
xgb_raw_accuracy_score = accuracy_score(y_raw_test, y_raw_pred_xgb)
print(xgb_raw_accuracy_score)

# visualize the confusion matrix of the XGBClassifier model on the raw dataset
xgb_raw_conf_matrix = confusion_matrix(y_raw_test, y_raw_pred_xgb)
print(xgb_raw_conf_matrix)
xgb_raw_conf_matrix_display = ConfusionMatrixDisplay(confusion_matrix = xgb_raw_conf_matrix)
xgb_raw_conf_matrix_display.plot()
plt.title("XGBClassifier model on the raw dataset")
plt.show()

# next we work with the RandomForestClassifier model on the raw data and see the accuracy
# create an instance of the RandomForestClassifier model
rf_model_raw = RandomForestClassifier(class_weight = "balanced", random_state = 21, verbose = 0)

# create the GridSearchCV for rf
rf_model_raw_grid = GridSearchCV(rf_model_raw, param_grid_rf, cv = kf, n_jobs = -1)

# fit the rf model with the raw train data
rf_model_raw_grid.fit(X_raw_train, y_raw_train) # type: ignore

# predict the X_raw_test with the rf model
y_raw_pred_rf = rf_model_raw_grid.predict(X_raw_test)

# print the classification report of the RandomForestClassifier model on the raw dataset
rf_raw_classification_report = classification_report(y_raw_test, y_raw_pred_rf)
print(rf_raw_classification_report)

# print the accuracy score
rf_raw_accuracy_score = accuracy_score(y_raw_test, y_raw_pred_rf)
print(rf_raw_accuracy_score)

# visualize the confusion matrix of the RandomForestClassifier model on the raw dataset
rf_raw_conf_matrix = confusion_matrix(y_raw_test, y_raw_pred_rf)
print(rf_raw_conf_matrix)
rf_raw_conf_matrix_display = ConfusionMatrixDisplay(confusion_matrix = rf_raw_conf_matrix)
rf_raw_conf_matrix_display.plot()
plt.title("RandomForestClassifier model on the raw dataset")
plt.show()

# Get probability predictions
rf_raw_probs = rf_model_raw_grid.predict_proba(X_raw_test)[:, 1]  # Probability of Class 1
xgb_raw_probs = xgb_model_raw_grid.predict_proba(X_raw_test)[:, 0]  # Probability of Class 0

# Define threshold-based model selection
def raw_combined_prediction(rf_prob, xgb_prob, threshold_1 = 0.5, threshold_0 = 0.9):
    """
    Selects Random Forest for Class 1 when confident and XGBoost for Class 0 when confident.
    Otherwise, uses weighted averaging or fallback logic.
    """
    if rf_prob > threshold_1:  # Random Forest strongly predicts Class 1
        return 1
    elif xgb_prob > threshold_0:  # XGBoost strongly predicts Class 0
        return 0
    else:
        return int(xgb_prob < 0.9)  # Fallback: Use XGBoost probabilities

# Apply decision logic
final_raw_preds = [raw_combined_prediction(rf, xgb) for rf, xgb in zip(rf_raw_probs, xgb_raw_probs)]
# final_preds contains the combined model predictions and y_test contains the actual labels

# Calculate accuracy of combined model
accuracy = accuracy_score(y_raw_test, final_raw_preds)
print("Accuracy of Combined Model:", accuracy)

# print the f1_score, roc_auc_score, precision and recall of the combined model
combined_precision = precision_score(y_raw_test, final_raw_preds)
combined_recall = recall_score(y_raw_test, final_raw_preds)
combined_f1_score = f1_score(y_raw_test, final_raw_preds)

print("combined Precision:", combined_precision)
print("combined Recall:", combined_recall)
print("combined f1_score:", combined_f1_score)

# visualize the confusion matrix of the combined model on the raw data
combined_conf_matrix = confusion_matrix(y_raw_test, final_raw_preds)
print(combined_conf_matrix)
combined_conf_matrix_display = ConfusionMatrixDisplay(confusion_matrix = combined_conf_matrix)
combined_conf_matrix_display.plot()
plt.title("Combined Prediction model on the raw dataset")
plt.show()

# now carryout preprocessing
# drop the Bankcrupt column since it is the target variable and every other column is the features variable
x_data = dataset_2.drop(["Bankrupt?"], axis = 1)
y_data = dataset_2["Bankrupt?"]

# Identify constant features
constant_features = [col for col in x_data.columns if x_data[col].nunique() == 1]

# Drop constant features
x_data.drop(columns = constant_features, inplace = True)

print(f"Dropped {len(constant_features)} constant features: {constant_features}")

# define a function to visualize only features with correlation greater than 0.85
def visualize_high_correlation(df, threshold=0.85):
    """
    Visualize only features with correlation ≥ threshold without showing correlation values.

    Parameters:
    - df: pandas DataFrame
    - threshold: float (default 0.85), correlation threshold

    Returns:
    - Displays correlation heatmap of highly correlated features (no values shown)
    """
    # Calculate correlation matrix
    corr_matrix = df.corr()

    # Find features with at least one correlation ≥ threshold
    high_corr_features = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                colname = corr_matrix.columns[i]
                high_corr_features.add(colname)
                rowname = corr_matrix.index[j]
                high_corr_features.add(rowname)

    # If no features meet threshold, inform user
    if not high_corr_features:
        print(f"No features found with correlation ≥ {threshold}")
        return

    # Create subset correlation matrix
    high_corr_df = df[list(high_corr_features)]
    high_corr_matrix = high_corr_df.corr()

    # Plot heatmap without values
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(high_corr_matrix, dtype=bool))
    sns.heatmap(high_corr_matrix, mask=mask, annot=False,  # Set annot=False to remove values
                cmap='coolwarm', vmin=-1, vmax=1,
                cbar_kws={"shrink": 0.75})
    plt.title(f"Features with Correlation ≥ {threshold}", pad=20, fontsize=14)
    plt.xticks(rotation=90, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.show()

# call the function
visualize_high_correlation(x_data, threshold = 0.85)

# Compute correlation matrix
correlation_matrix = x_data.corr()

# Set correlation threshold
threshold = 0.85

# Identify highly correlated feature pairs
correlated_features = set()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold: # type: ignore
            colname = correlation_matrix.columns[i]  # Pick one column from the pair
            correlated_features.add(colname)  # Add to removal list

# Drop one column from each pair
x_data.drop(columns = correlated_features, inplace = True)

# print(f"Removed {len(correlated_features)} highly correlated features.")

# Calculate skewness for numerical columns
skewness = x_data.skew()

# Identify extremely skewed columns (|skew| > 2)
extremely_skewed_cols = skewness[skewness.abs() > 2].index.tolist()

# for col in extremely_skewed_cols:
#     fig = plt.figure(figsize = (12, 4))
#     sns.histplot(x=x_data[col], kde=True)
#     plt.title(f"{col} (Skew: {skewness[col]:.2f})")
#     plt.ylabel("Count")
#     plt.tight_layout()
#     plt.show()

# print("Skewness Before Transformation:\n", x_data[extremely_skewed_cols].skew())

# Apply Quantile Transformation only to extreme skewed columns
qt = QuantileTransformer(output_distribution='normal')
x_data[extremely_skewed_cols] = qt.fit_transform(x_data[extremely_skewed_cols])

# print("Skewness After Transformation:\n", x_data[extremely_skewed_cols].skew())

# select the features variable from the dataset and convert to a numpy array
X = x_data.values

# select the target variable from the dataset and convert to a numpy array
y = y_data.values

# split the data into 70 percent train and 30 percent test, using the features and the target variable of the raw data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 21, stratify = y) # type: ignore

scale_pos_weight = round(sum(y == 0) / sum(y == 1)) + 1 # type: ignore

# the XGBClassifier model
# create an instance of the XGBClassifier model
xgb_model = XGBClassifier(scale_pos_weight = scale_pos_weight, random_state = 21)

# create the RandomizedSearchCV for xgb
xgb_model_grid = RandomizedSearchCV(xgb_model, param_grid_xgb, cv = kf, n_jobs = -1) # type: ignore

# fit the xgb model with the train data
xgb_model_grid.fit(X_train, y_train)

# Get the best trained model
best_model = xgb_model_grid.best_estimator_

# Get feature importances
feature_importances = best_model.feature_importances_ # type: ignore

# Convert to DataFrame
x_train = pd.DataFrame(X_train)

# Create a DataFrame for visualization
features_df = pd.DataFrame({
    "Feature": x_train.columns,
    "Importance": feature_importances
})

# Sort by importance (descending)
features_df = features_df.sort_values("Importance", ascending = False)

features_name = []
for num in features_df.index:
    features_name.append(x_data.columns[num])

features_df["features_name"] = features_name

# Get top 5 features
top_5_features = features_df.head(5)
print(top_5_features)

# Get top 10 features
top_10_features = features_df.head(10)

# Create visualization for the top 10 features
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='features_name', data=top_10_features, palette='viridis')

# Customize the plot
plt.title('Top 10 Most Important Features', fontsize=16)
plt.xlabel('Feature Importance Score', fontsize=12)
plt.ylabel('Feature Name', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Add value labels
for index, value in enumerate(top_10_features['Importance']):
    plt.text(value, index, f'{value:.4f}', va='center')

plt.tight_layout()
plt.show()

# predict the X_test with the xgb model
y_pred_xgb = xgb_model_grid.predict(X_test)

# get the probabilites for the xgb_model
y_probs_xgb = xgb_model_grid.predict_proba(X_test)

# print the classification report of the XGBClassifier model on the preprocessed dataset
xgb_classification_report = classification_report(y_test, y_pred_xgb)
print(xgb_classification_report)

# print the accuracy score
xgb_accuracy_score = accuracy_score(y_test, y_pred_xgb)
print(xgb_accuracy_score)

# print the f1_score, roc_auc_score, precision and recall of the randon forest model
xgb_precision = precision_score(y_test, y_pred_xgb)
xgb_recall = recall_score(y_test, y_pred_xgb)
xgb_f1_score = f1_score(y_test, y_pred_xgb)
xgb_roc_auc_score = roc_auc_score(y_test, y_probs_xgb[:, 1])

print("XGBoost Precision:", xgb_precision)
print("XGBoost Recall:", xgb_recall)
print("XGBoost f1_score:", xgb_f1_score)
print("XGBoost roc_auc_score:", xgb_roc_auc_score)

# visualize the confusion matrix of the XGBClassifier model on the preprocessed dataset
xgb_conf_matrix = confusion_matrix(y_test, y_pred_xgb)
print(xgb_conf_matrix)
xgb_conf_matrix_display = ConfusionMatrixDisplay(confusion_matrix = xgb_conf_matrix)
xgb_conf_matrix_display.plot()
plt.title("XGBoost Classifier model on the processed dataset")
plt.show()

# the RandomForestClassifier model
# create an instance of the RandomForestClassifier model
rf_model = RandomForestClassifier(class_weight = "balanced", random_state = 21)

# create the GridSearchCV for rf
rf_model_grid = GridSearchCV(rf_model, param_grid_rf, cv = kf, n_jobs = -1)

# fit the rf model with the train data
rf_model_grid.fit(X_train, y_train) # type: ignore

# predict the X_test with the rf model
y_pred_rf = rf_model_grid.predict(X_test)

# get the probabilites for the rf_model
y_probs_rf = rf_model_grid.predict_proba(X_test)

# print the classification report of the RandomForestClassifier model on the preprocessed dataset
rf_classification_report = classification_report(y_test, y_pred_rf)
print(rf_classification_report)

# print the accuracy score
rf_accuracy_score = accuracy_score(y_test, y_pred_rf)
print(rf_accuracy_score)

# print the f1_score, roc_auc_score, precision and recall of the randon forest model
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1_score = f1_score(y_test, y_pred_rf)
rf_roc_auc_score = roc_auc_score(y_test, y_probs_rf[:, 1])

print("RandomForest Precision:", rf_precision)
print("RandomForest Recall:", rf_recall)
print("RandomForest f1_score:", rf_f1_score)
print("RandomForest roc_auc_score:", rf_roc_auc_score)

# visualize the confusion matrix of the RandomForestClassifier model on the preprocessed dataset
rf_conf_matrix = confusion_matrix(y_test, y_pred_rf)
print(rf_conf_matrix)
rf_conf_matrix_display = ConfusionMatrixDisplay(confusion_matrix = rf_conf_matrix)
rf_conf_matrix_display.plot()
plt.title("RandomForestClassifier model on the processed dataset")
plt.show()

# Get probability predictions
rf_probs = rf_model_grid.predict_proba(X_test)[:, 1]  # Probability of Class 1
xgb_probs = xgb_model_grid.predict_proba(X_test)[:, 0]  # Probability of Class 0

# Define threshold-based model selection
def combined_prediction(rf_prob, xgb_prob, threshold_1 = 0.5, threshold_0 = 0.9):
    """
    Selects Random Forest for Class 1 when confident and XGBoost for Class 0 when confident.
    Otherwise, uses weighted averaging or fallback logic.
    """
    if rf_prob > threshold_1:  # Random Forest strongly predicts Class 1
        return 1
    elif xgb_prob > threshold_0:  # XGBoost strongly predicts Class 0
        return 0
    else:
        return int(xgb_prob < 0.9)  # Fallback: Use XGBoost probabilities

# Apply decision logic
final_preds = [combined_prediction(rf, xgb) for rf, xgb in zip(rf_probs, xgb_probs)]
# final_preds contains the combined model predictions and y_test contains the actual labels

# Calculate accuracy of combined model
accuracy = accuracy_score(y_test, final_preds)
print("Accuracy of Combined Model:", accuracy)

# print the f1_score, roc_auc_score, precision and recall of the combined model
combined_precision = precision_score(y_test, final_preds)
combined_recall = recall_score(y_test, final_preds)
combined_f1_score = f1_score(y_test, final_preds)

print("combined Precision:", combined_precision)
print("combined Recall:", combined_recall)
print("combined f1_score:", combined_f1_score)

# visualize the confusion matrix of the combined model
combined_conf_matrix = confusion_matrix(y_test, final_preds)
print(combined_conf_matrix)
combined_conf_matrix_display = ConfusionMatrixDisplay(confusion_matrix = combined_conf_matrix)
combined_conf_matrix_display.plot()
plt.title("Combined Prediction model on the processed dataset")
plt.show()

# Get the best trained model
best_model = rf_model_grid.best_estimator_

# Get predicted probabilities for the positive class (class 1)
y_probs = best_model.predict_proba(X_test)[:, 1]  # type: ignore

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Create ROC curve plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')

# Customize the plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)

# Highlight optimal threshold (Youden's J statistic)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red',
            label=f'Optimal Threshold ({optimal_threshold:.2f})')

plt.show()

# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
avg_precision = average_precision_score(y_test, y_probs)

# Create the plot
plt.figure(figsize=(8, 6))

# Plot the precision-recall curve
plt.plot(recall, precision, color='blue', lw=2,
         label=f'Precision-Recall (AP = {avg_precision:.2f})')

# Plot the no-skill line (proportion of positive cases)
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='red',
         label='No Skill')

# Highlight the optimal threshold point (F1-score maximization)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
optimal_idx = np.argmax(f1_scores)
plt.scatter(recall[optimal_idx], precision[optimal_idx], marker='o', color='black',
            label=f'Optimal Threshold ({thresholds[optimal_idx]:.2f}, F1={f1_scores[optimal_idx]:.2f})')

# Customize the plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall (True Positive Rate)', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14)
plt.legend(loc="lower left", fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Define your scoring metric
scoring_metric = 'f1' ]

# Calculate learning curve
train_sizes, train_scores, val_scores = learning_curve( # type: ignore
    estimator=best_model,
    X=X_train,
    y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),  # 10 different training set sizes
    cv=5,  # 5-fold cross-validation
    scoring=scoring_metric,
    n_jobs=-1,  # use all available cores
    shuffle=True,
    random_state=21
)

# Calculate mean and standard deviation
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)

# Create the plot
plt.figure(figsize=(10, 6))
plt.title(f"Learning Curve ({scoring_metric.upper()})", fontsize=14)
plt.xlabel("Training Examples", fontsize=12)
plt.ylabel(scoring_metric.upper(), fontsize=12)

# Plot the learning curves
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                 val_scores_mean + val_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         label="Training score")
plt.plot(train_sizes, val_scores_mean, 'o-', color="g",
         label="Cross-validation score")

# Customize the plot
plt.grid(True, alpha=0.3)
plt.legend(loc="best", fontsize=12)
plt.ylim(0, 1.1)  # Adjust based on your metric range

# Add text for final scores
final_train_score = train_scores_mean[-1]
final_val_score = val_scores_mean[-1]
plt.text(0.7, 0.2, f'Final Training {scoring_metric}: {final_train_score:.3f}',
         transform=plt.gca().transAxes, fontsize=10)
plt.text(0.7, 0.1, f'Final Validation {scoring_metric}: {final_val_score:.3f}',
         transform=plt.gca().transAxes, fontsize=10)

plt.tight_layout()
plt.show()
