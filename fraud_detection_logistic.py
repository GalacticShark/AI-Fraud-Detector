# Credit Card Fraud Detection using Machine Learning

# 1. Importing the required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from imblearn.over_sampling import RandomOverSampler

# 2. Reading the dataset
data = pd.read_csv('creditcard.csv')

# 3. Exploratory data analysis
print(data.head())
print(data.describe())
print(data.info())

# Check class imbalance
print(data['Class'].value_counts())
plt.figure(figsize=(10, 8))
class_counts = data['Class'].value_counts()
plt.bar(['Not Fraud (0)', 'Fraud (1)'], class_counts.values, color=['#3498db', '#e74c3c'], width=0.6)
plt.title('Class Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Class', fontsize=12)
plt.ylabel('Count', fontsize=12)
# Add count labels on bars
for i, v in enumerate(class_counts.values):
    plt.text(i, v + 5000, str(v), ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n[SAVED] Class distribution plot saved as 'class_distribution.png'")

# 4. Splitting the dataset into features and labels
X = data.drop('Class', axis=1)
y = data['Class']

# 5. Oversampling the minority class using RandomOverSampler
oversampler = RandomOverSampler()
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# 6. Scaling the features using StandardScaler
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

# 7. Splitting the resampled dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# 8. Creating the logistic regression model
logistic = LogisticRegression()

# 9. Setting up the GridSearchCV to optimize hyperparameters
params = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
grid_search = GridSearchCV(logistic, params, cv=5)

# 10. Training the model
print("\nTraining the model...")
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")

# 11. Predicting the labels for the testing set
y_pred = grid_search.predict(X_test)

# 12. Evaluating the model performance
confusion = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix:\n", confusion)

report = classification_report(y_test, y_pred)
print("\nClassification report:\n", report)

accuracy = grid_search.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")

# 13. Plotting the ROC curve
y_prob = grid_search.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f"ROC curve (area = {auc:.2f})", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n[SAVED] ROC curve saved as 'roc_curve.png'")

# 14. Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, cmap='Blues', annot=True, fmt='g', cbar=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("[SAVED] Confusion matrix saved as 'confusion_matrix.png'")

print("\n" + "="*50)
print("All plots have been saved! Check your folder for:")
print("  - class_distribution.png")
print("  - roc_curve.png")
print("  - confusion_matrix.png")
print("="*50)