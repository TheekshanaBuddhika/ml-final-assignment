import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Step 1: Load the dataset
print("Step 1: Loading the dataset...")
url = "/Users/savindajayasekara/WorkZone/ML/test/bank.csv"
df = pd.read_csv(url, sep=';')

# Display basic information about the dataset
print("\nDataset shape:", df.shape)
print("\nFirst 5 rows of the dataset:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Check the distribution of the target variable
print("\nDistribution of target variable:")
print(df['y'].value_counts())
print(df['y'].value_counts(normalize=True) * 100)

# Visualize the distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='y', data=df)
plt.title('Distribution of Target Variable')
plt.savefig('target_distribution.png')
plt.close()

# Step 2: Preprocess the dataset
print("\nStep 2: Preprocessing the dataset...")

# A. Handle Missing Values and Outliers
# Check for 'unknown' values in categorical columns
print("\nChecking for 'unknown' values in categorical columns:")
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
for col in categorical_cols:
    if 'unknown' in df[col].unique():
        print(f"{col}: {df[col].value_counts()['unknown']} unknown values")

# Replace 'unknown' with NaN for proper handling
for col in categorical_cols:
    if 'unknown' in df[col].unique():
        df[col] = df[col].replace('unknown', np.nan)

# Check for outliers in numerical columns
print("\nChecking for outliers in numerical columns:")
numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# Create box plots for numerical columns to visualize outliers
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(3, 3, i+1)
    sns.boxplot(x=df[col])
    plt.title(col)
plt.tight_layout()
plt.savefig('outliers_boxplot.png')
plt.close()

# B. Feature Coding for Categorical Variables
# Convert target variable to binary (0/1)
print("\nConverting target variable to binary...")
df['y'] = df['y'].map({'no': 0, 'yes': 1})

# C. Scale/Standardize Features
# Identify categorical and numerical columns
print("\nIdentifying categorical and numerical columns...")
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
                   'contact', 'month', 'poutcome']
numerical_cols = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']

# Note: Removing 'duration' as mentioned in the dataset description
print("\nRemoving 'duration' as it's not suitable for a realistic predictive model...")

# Split the data into features and target
X = df.drop(['y', 'duration'], axis=1)
y = df['y']

# Split the data into training and testing sets (80% train, 20% test)
print("\nSplitting data into training and testing sets (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Create preprocessing pipelines for both numerical and categorical data
print("\nCreating preprocessing pipelines...")

# Numerical pipeline
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Step 3: Train the models
print("\nStep 3: Training the models...")

# A. Support Vector Machine (SVM)
print("\nTraining SVM model...")
svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(probability=True, random_state=42))
])

svm_pipeline.fit(X_train, y_train)

# Evaluate SVM model
print("\nEvaluating SVM model...")
y_pred_svm = svm_pipeline.predict(X_test)
print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm))
print("\nSVM Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))
print(f"\nSVM Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")

# B. Logistic Regression
print("\nTraining Logistic Regression model...")
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

lr_pipeline.fit(X_train, y_train)

# Evaluate Logistic Regression model
print("\nEvaluating Logistic Regression model...")
y_pred_lr = lr_pipeline.predict(X_test)
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))
print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))
print(f"\nLogistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")

# Step 4: Save the models as pickle files
print("\nStep 4: Saving the models as pickle files...")

# Save the SVM model
with open('svm_model.pkl', 'wb') as file:
    pickle.dump(svm_pipeline, file)
print("SVM model saved as 'svm_model.pkl'")

# Save the Logistic Regression model
with open('lr_model.pkl', 'wb') as file:
    pickle.dump(lr_pipeline, file)
print("Logistic Regression model saved as 'lr_model.pkl'")

# Step 5: Demonstrate how to load and use the saved models
print("\nStep 5: Demonstrating how to load and use the saved models...")

# Load the models
with open('svm_model.pkl', 'rb') as file:
    loaded_svm_model = pickle.load(file)

with open('lr_model.pkl', 'rb') as file:
    loaded_lr_model = pickle.load(file)

# Create a sample input
print("\nCreating a sample input...")
sample_input = pd.DataFrame({
    'age': [35],
    'job': ['management'],
    'marital': ['married'],
    'education': ['tertiary'],
    'default': ['no'],
    'balance': [1500],
    'housing': ['yes'],
    'loan': ['no'],
    'contact': ['cellular'],
    'day': [15],
    'month': ['may'],
    'campaign': [1],
    'pdays': [999],
    'previous': [0],
    'poutcome': ['nonexistent']
})

# Make predictions using the loaded models
print("\nMaking predictions using the loaded models...")
svm_prediction = loaded_svm_model.predict(sample_input)
lr_prediction = loaded_lr_model.predict(sample_input)

print(f"SVM Prediction: {'Yes' if svm_prediction[0] == 1 else 'No'}")
print(f"Logistic Regression Prediction: {'Yes' if lr_prediction[0] == 1 else 'No'}")

print("\nDone!")