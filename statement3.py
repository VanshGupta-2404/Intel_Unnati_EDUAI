import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import re
import textstat

class ContentFilteringSystem:
    def __init__(self):
        """
        Initialize the Content Filtering System with ML models for content classification
        """
        # Feature extraction methods
        self.feature_extractors = {
            'readability_score': self.get_readability_score,
            'complexity_score': self.get_complexity_score,
            'word_count': self.get_word_count,
            'sentence_length': self.get_average_sentence_length
        }

    def get_readability_score(self, text):
        """
        Calculate Flesch-Kincaid readability score
        
        Args:
            text (str): Input text
        
        Returns:
            float: Readability score
        """
        try:
            return textstat.flesch_kincaid_grade(text)
        except:
            return 0

    def get_complexity_score(self, text):
        """
        Calculate text complexity based on vocabulary and syntax
        
        Args:
            text (str): Input text
        
        Returns:
            float: Complexity score
        """
        # Count of complex words (more than 3 syllables)
        complex_words = len([word for word in text.split() if textstat.syllable_count(word) > 3])
        total_words = len(text.split())
        return (complex_words / total_words) * 100 if total_words > 0 else 0

    def get_word_count(self, text):
        """
        Count total words in the text
        
        Args:
            text (str): Input text
        
        Returns:
            int: Number of words
        """
        return len(text.split())

    def get_average_sentence_length(self, text):
        """
        Calculate average sentence length
        
        Args:
            text (str): Input text
        
        Returns:
            float: Average sentence length
        """
        sentences = re.split(r'[.!?]+', text)
        sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence.strip()]
        return np.mean(sentence_lengths) if sentence_lengths else 0

    def extract_features(self, text):
        """
        Extract features from the text
        
        Args:
            text (str): Input text
        
        Returns:
            dict: Extracted features
        """
        return {
            name: extractor(text) 
            for name, extractor in self.feature_extractors.items()
        }

    def prepare_training_data(self, texts, grade_levels):
        """
        Prepare training data for the ML model
        
        Args:
            texts (list): List of text samples
            grade_levels (list): Corresponding grade levels
        
        Returns:
            tuple: Prepared features and encoded labels
        """
        # Extract features
        features = [self.extract_features(text) for text in texts]
        
        # Convert to DataFrame
        df = pd.DataFrame(features)
        
        # Encode grade levels
        le = LabelEncoder()
        y_encoded = le.fit_transform(grade_levels)
        
        return df, y_encoded, le

    def create_grade_level_classifier(self):
        """
        Create a machine learning pipeline for grade-level classification
        
        Returns:
            Pipeline: Scikit-learn classification pipeline
        """
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                max_depth=10
            ))
        ])
        return pipeline

    def train_content_filter_model(self, texts, grade_levels):
        """
        Train a model to classify content appropriateness
        
        Args:
            texts (list): Training text samples
            grade_levels (list): Corresponding grade levels
        
        Returns:
            dict: Trained model and related objects
        """
        # Prepare training data
        X, y, label_encoder = self.prepare_training_data(texts, grade_levels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create and train pipeline
        pipeline = self.create_grade_level_classifier()
        pipeline.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = pipeline.predict(X_test)
        
        # Get unique labels actually present in the test set
        unique_labels = np.unique(y_test)
        unique_target_names = label_encoder.inverse_transform(unique_labels)
        
        # Compute accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"📊 Model Accuracy: {accuracy:.2f}")
        
        # Print classification report with only present labels
        print("\nContent Classification Report:")
        print(classification_report(
            y_test, 
            y_pred, 
            labels=unique_labels,
            target_names=unique_target_names
        ))
        
        return {
            'model': pipeline,
            'label_encoder': label_encoder
        }

    def recommend_content_level(self, text, trained_model):
        """
        Recommend appropriate content level for a given text
        
        Args:
            text (str): Input text
            trained_model (dict): Trained model dictionary
        
        Returns:
            str: Recommended grade level
        """
        # Extract features
        features = pd.DataFrame([self.extract_features(text)])
        
        # Predict grade level
        predicted_level_index = trained_model['model'].predict(features)[0]
        recommended_level = trained_model['label_encoder'].inverse_transform([predicted_level_index])[0]
        
        return recommended_level

    def filter_content(self, text, trained_model, allowed_grades):
        """
        Filter content based on allowed grade levels
        
        Args:
            text (str): Input text
            trained_model (dict): Trained model dictionary
            allowed_grades (list): List of allowed grade levels
        
        Returns:
            dict: Filtering results
        """
        # Predict content level
        recommended_level = self.recommend_content_level(text, trained_model)
        
        # Check if content is appropriate
        is_appropriate = recommended_level in allowed_grades
        
        return {
            'original_text': text,
            'recommended_level': recommended_level,
            'is_appropriate': is_appropriate,
            'features': self.extract_features(text)
        }

def main():
    # Expanded training data with more diverse examples
    training_texts = [
        "The cat sat on the mat.",  # Simple text
        "Mammals are warm-blooded animals that feed their young with milk.",  # Simple to medium
        "Photosynthesis is the process by which green plants transform light energy into chemical energy.",  # Medium complexity
        "Electricity flows through conductors and powers various electronic devices.",  # Medium complexity
        "The mitochondria is known as the powerhouse of the cell, generating ATP through cellular respiration.",  # Advanced text
        "Quantum mechanics describes nature at the smallest scales of energy levels of atoms and subatomic particles.",  # Advanced text
    ]
    
    training_grades = [
        '2nd Grade',  # Simple text
        '3rd Grade',  # Simple to medium
        '6th Grade',  # Medium complexity
        '7th Grade',  # Medium complexity
        '10th Grade',  # Advanced text
        '11th Grade'  # Advanced text
    ]
    
    # Initialize Content Filtering System
    content_filter = ContentFilteringSystem()
    
    # Train the model
    trained_model = content_filter.train_content_filter_model(
        training_texts, 
        training_grades
    )
    
    # Example texts to filter
    test_texts = [
        "Elephants are large mammals that live in Africa and Asia.",
        "The chemical composition of water is H2O, composed of two hydrogen atoms and one oxygen atom.",
        "Calculus involves the study of continuous change through derivatives and integrals."
    ]
    
    # Allowed grade levels for filtering
    allowed_grades = ['3rd Grade', '4th Grade', '5th Grade', '6th Grade']
    
    # Filter and recommend content levels
    print("\n🔍 Content Filtering Results:")
    for text in test_texts:
        result = content_filter.filter_content(text, trained_model, allowed_grades)
        print("\nText Analysis:")
        print(f"Original Text: {result['original_text']}")
        print(f"Recommended Level: {result['recommended_level']}")
        print(f"Appropriate for Grades: {allowed_grades}")
        print(f"Is Appropriate: {result['is_appropriate']}")
        print("Features:")
        for feature, value in result['features'].items():
            print(f"  - {feature}: {value}")

if __name__ == "__main__":
    main()