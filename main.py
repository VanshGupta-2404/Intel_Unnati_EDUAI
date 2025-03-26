import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("indian_student_data.csv")  
print("Dataset Columns:", df.columns)

# Strip any spaces from column names (if any)
df.columns = df.columns.str.strip()

# Rename columns to match expected format
df.rename(columns={
    "Study Time per Day (hrs)": "study_hours",
    "IQ of Student": "iq_level",
    "Material Level": "difficulty_level",
    "Assessment Score": "assessment_score"
}, inplace=True)

# Verify dataset structure after renaming
print(df.head())

# Select relevant columns
expected_columns = {'study_hours', 'iq_level', 'difficulty_level'}

# Check if all required columns exist
missing_cols = expected_columns - set(df.columns)
if missing_cols:
    raise KeyError(f"Missing columns in dataset: {missing_cols}")

# Drop rows with missing values (if any)
df = df.dropna()

# Define features and target variable
X = df[['study_hours', 'iq_level', 'difficulty_level']]
y = df['assessment_score']  # Target variable

# Convert categorical data (if applicable)
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training with tuned hyperparameters
model = RandomForestRegressor(
    n_estimators=200,  # Increased trees for better performance
    max_depth=10,  # Limit depth to prevent overfitting
    min_samples_split=5,  # Minimum samples to split
    min_samples_leaf=2,  # Minimum samples in leaf node
    random_state=42
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Model Performance:")
print(f"➡ Mean Absolute Error: {mae:.4f}")
print(f"➡ Mean Squared Error: {mse:.4f}")
print(f"➡ R² Score: {r2:.4f}")

# Function to decide student promotion
def should_promote(predicted_score, threshold=60):
    return "Promoted ✅" if predicted_score >= threshold else "Not Promoted ❌"

# Example student prediction
sample_student = np.array([[5, 129, 2]])  # 5 study hours, IQ 129, difficulty level 2
predicted_score = model.predict(sample_student.reshape(1, -1))[0]

print(f"\n📌 **Example Student Prediction:**")
print(f"🎯 Predicted Score: {predicted_score:.2f}")
print(f"📢 Decision: {should_promote(predicted_score)}")
