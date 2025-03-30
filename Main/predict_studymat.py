import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load dataset
def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        df.rename(columns={
            "Study Time per Day (hrs)": "study_hours",
            "IQ of Student": "iq_level",
            "Material Level": "difficulty_level",
            "Assessment Score": "assessment_score",
            "Material Name": "material_name"
        }, inplace=True)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Prepare features and target
def prepare_data(df):
    X = df[['study_hours', 'iq_level', 'difficulty_level', 'assessment_score']]
    y = df['material_name']
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return X, y_encoded, label_encoder

# Create ML pipeline
def create_ml_pipeline():
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['study_hours', 'iq_level', 'difficulty_level', 'assessment_score'])
    ])
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=3, random_state=42))
    ])
    return pipeline

# Train and evaluate model
def train_and_evaluate_model(file_path):
    df = load_and_preprocess_data(file_path)
    if df is None:
        return None
    X, y, label_encoder = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = create_ml_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    print(f"✅ Model Accuracy: {accuracy:.4f}\n")
    print("📊 Classification Report:\n", classification_report(y_test_labels, y_pred_labels))

    # Confusion Matrix
    plot_confusion_matrix(y_test_labels, y_pred_labels, label_encoder.classes_)
    
    # Feature Importance
    plot_feature_importance(pipeline, X.columns)

    return pipeline, label_encoder

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_labels):
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")  # Save instead of show
    print("✅ Confusion Matrix saved as confusion_matrix.png\n")

# Plot feature importance
def plot_feature_importance(pipeline, features):
    importances = pipeline.named_steps['classifier'].feature_importances_
    plt.figure(figsize=(7, 5))
    sns.barplot(x=importances, y=features, palette="viridis", hue=None)  # Fix warning
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.title("Feature Importance")
    plt.savefig("feature_importance.png")  # Save instead of show
    print("✅ Feature Importance graph saved as feature_importance.png\n")

# Recommend study material
def recommend_material(pipeline, label_encoder, study_hours, iq_level, difficulty_level, assessment_score):
    input_data = pd.DataFrame([{
        'study_hours': study_hours,
        'iq_level': iq_level,
        'difficulty_level': difficulty_level,
        'assessment_score': assessment_score
    }])
    predicted_index = pipeline.predict(input_data)[0]
    return label_encoder.inverse_transform([predicted_index])[0]

# Main execution
def main():
    result = train_and_evaluate_model("indian_student_data.csv")
    if result:
        pipeline, label_encoder = result
        example_material = recommend_material(pipeline, label_encoder, study_hours=3.5, iq_level=110, difficulty_level=1, assessment_score=70)
        print(f"\n📚 Recommended Study Material: {example_material}")

# Run
if __name__ == "__main__":
    main()
