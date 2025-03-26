import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
file_path = "indian_student_data.csv"
# Load dataset
def load_and_preprocess_data(file_path):
    try:
        # Load dataset
        df = pd.read_csv(file_path)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        df.rename(columns={
            "Study Time per Day (hrs)": "study_hours",
            "IQ of Student": "iq_level",
            "Material Level": "difficulty_level",
            "Assessment Score": "assessment_score",
            "Material Name": "material_name"
        }, inplace=True)
        
        # Drop missing values
        df.dropna(inplace=True)
        
        return df
    except FileNotFoundError:
        print(f"Error: File not found. Please check the file path.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

# Prepare features and target
def prepare_data(df):
    # Separate features and target
    X = df[['study_hours', 'iq_level', 'difficulty_level', 'assessment_score']]
    y = df['material_name']
    
    # Encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    return X, y_encoded, label_encoder

# Create machine learning pipeline
def create_ml_pipeline():
    # Numerical features preprocessing
    numeric_features = ['study_hours', 'iq_level', 'difficulty_level', 'assessment_score']
    
    # Preprocessing for numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features)
        ])
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=300, 
            max_depth=20, 
            min_samples_split=3, 
            random_state=42
        ))
    ])
    
    return pipeline

# Main training and evaluation function
def train_and_evaluate_model(file_path):
    # Load and preprocess data
    df = load_and_preprocess_data(file_path)
    if df is None:
        return None
    
    # Prepare features and target
    X, y, label_encoder = prepare_data(df)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train the pipeline
    pipeline = create_ml_pipeline()
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Convert back to original labels
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    print(f"✅ Model Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\n📊 Classification Report:")
    print(classification_report(y_test_labels, y_pred_labels))
    
    return pipeline, label_encoder

# Function to recommend material
def recommend_material(pipeline, label_encoder, study_hours, iq_level, difficulty_level, assessment_score):
    # Prepare input data
    input_data = pd.DataFrame([{
        'study_hours': study_hours, 
        'iq_level': iq_level, 
        'difficulty_level': difficulty_level, 
        'assessment_score': assessment_score
    }])
    
    # Make prediction
    predicted_index = pipeline.predict(input_data)[0]
    
    # Convert back to original label
    recommended_material = label_encoder.inverse_transform([predicted_index])[0]
    
    return recommended_material

# Main execution
def main():
    # Train the model
    result = train_and_evaluate_model("indian_student_data.csv")
    
    if result:
        pipeline, label_encoder = result
        
        # Example prediction
        example_material = recommend_material(
            pipeline, 
            label_encoder, 
            study_hours=3.5, 
            iq_level=110, 
            difficulty_level=1, 
            assessment_score=70
        )
        print(f"\n📚 Recommended Study Material: {example_material}")

# Run the main function
if __name__ == "__main__":
    main()