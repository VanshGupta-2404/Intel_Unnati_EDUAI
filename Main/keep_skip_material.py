import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import re
import textstat
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from collections import Counter

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
            'sentence_length': self.get_average_sentence_length,
            'avg_word_length': self.get_average_word_length,
            'unique_words_ratio': self.get_unique_words_ratio,
            'special_chars_ratio': self.get_special_chars_ratio
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
    
    def get_average_word_length(self, text):
        """
        Calculate average word length
        
        Args:
            text (str): Input text
        
        Returns:
            float: Average word length
        """
        words = [word for word in text.split() if word.strip()]
        word_lengths = [len(word) for word in words]
        return np.mean(word_lengths) if word_lengths else 0
    
    def get_unique_words_ratio(self, text):
        """
        Calculate ratio of unique words to total words
        
        Args:
            text (str): Input text
        
        Returns:
            float: Unique words ratio
        """
        words = [word.lower() for word in text.split() if word.strip()]
        unique_words = set(words)
        return len(unique_words) / len(words) if words else 0
    
    def get_special_chars_ratio(self, text):
        """
        Calculate ratio of special characters to total characters
        
        Args:
            text (str): Input text
        
        Returns:
            float: Special characters ratio
        """
        special_chars = sum(1 for char in text if not char.isalnum() and not char.isspace())
        total_chars = len(text)
        return special_chars / total_chars if total_chars > 0 else 0

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
        
        # Add TF-IDF features if we have enough samples
        if len(texts) >= 5:  # Only use TF-IDF if we have enough samples
            tfidf_vectorizer = TfidfVectorizer(max_features=min(10, len(texts) - 1), stop_words='english')
            tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(), 
                columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
            )
            # Combine all features
            combined_df = pd.concat([df, tfidf_df], axis=1)
        else:
            combined_df = df
        
        # Encode grade levels
        le = LabelEncoder()
        y_encoded = le.fit_transform(grade_levels)
        
        return combined_df, y_encoded, le

    def augment_minority_classes(self, X, y, min_samples=2):
        """
        Augment minority classes to have at least min_samples
        
        Args:
            X (DataFrame): Features
            y (array): Labels
            min_samples (int): Minimum number of samples per class
            
        Returns:
            tuple: Augmented features and labels
        """
        # Count samples per class
        class_counts = Counter(y)
        
        # Identify minority classes
        minority_classes = [cls for cls, count in class_counts.items() if count < min_samples]
        
        if not minority_classes:
            return X, y
        
        X_aug = X.copy()
        y_aug = y.copy()
        
        for cls in minority_classes:
            # Get indices of the minority class
            indices = np.where(y == cls)[0]
            
            # Calculate how many samples to add
            n_to_add = min_samples - len(indices)
            
            # Add samples with small random noise
            for _ in range(n_to_add):
                # Pick a random sample from the class
                idx = np.random.choice(indices)
                
                # Create a new sample with small random variations
                new_sample = X.iloc[idx].copy()
                
                # Add small noise to numerical features
                for col in new_sample.index:
                    if isinstance(new_sample[col], (int, float)):
                        # Add noise (±5%)
                        noise = np.random.uniform(-0.05, 0.05) * new_sample[col]
                        new_sample[col] += noise
                
                # Add the new sample
                X_aug = pd.concat([X_aug, pd.DataFrame([new_sample])], ignore_index=True)
                y_aug = np.append(y_aug, cls)
        
        return X_aug, y_aug

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
                max_depth=10,
                class_weight='balanced'
            ))
        ])
        return pipeline

    def train_with_cross_validation(self, X, y, n_splits=5):
        """
        Train model with cross-validation when dataset is too small for train_test_split
        
        Args:
            X (DataFrame): Features
            y (array): Labels
            n_splits (int): Number of cross-validation splits
            
        Returns:
            dict: Trained model and performance metrics
        """
        # Create a basic pipeline
        pipeline = self.create_grade_level_classifier()
        
        # Prepare cross-validation
        kf = KFold(n_splits=min(n_splits, len(X)), shuffle=True, random_state=42)
        
        # Store predictions and scores
        all_predictions = []
        all_true_values = []
        fold_accuracies = []
        
        # Cross-validation
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Train the model
            pipeline.fit(X_train, y_train)
            
            # Predict
            y_pred = pipeline.predict(X_test)
            
            # Store results
            all_predictions.extend(y_pred)
            all_true_values.extend(y_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            fold_accuracies.append(accuracy)
        
        # Train final model on all data
        pipeline.fit(X, y)
        
        # Calculate overall accuracy
        overall_accuracy = accuracy_score(all_true_values, all_predictions)
        
        return {
            'model': pipeline,
            'accuracy': overall_accuracy,
            'fold_accuracies': fold_accuracies
        }

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
        
        # Check class distribution
        class_counts = Counter(y)
        print(f"\n📊 Class distribution: {class_counts}")
        
        # Augment minority classes if needed
        min_samples_per_class = 2
        if min(class_counts.values()) < min_samples_per_class:
            print(f"⚠️ Detected classes with less than {min_samples_per_class} samples")
            print("📈 Augmenting minority classes...")
            X, y = self.augment_minority_classes(X, y, min_samples_per_class)
            print(f"👍 After augmentation: {Counter(y)}")
        
        # Check if dataset is too small for train_test_split
        unique_classes = len(set(y))
        if len(texts) < unique_classes * 2:
            print("⚠️ Dataset too small for train_test_split, using cross-validation instead")
            
            # Train with cross-validation
            results = self.train_with_cross_validation(X, y)
            
            pipeline = results['model']
            accuracy = results['accuracy']
            
            print(f"📊 Model Accuracy: {accuracy:.2f}")
            print(f"📊 Fold Accuracies: {[f'{acc:.2f}' for acc in results['fold_accuracies']]}")
        else:
            # Use train_test_split if we have enough data
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Create and train pipeline
                pipeline = self.create_grade_level_classifier()
                pipeline.fit(X_train, y_train)
                
                # Predict and evaluate
                y_pred = pipeline.predict(X_test)
                
                # Compute accuracy
                accuracy = accuracy_score(y_test, y_pred)
                print(f"📊 Model Accuracy: {accuracy:.2f}")
                
                # Print classification report
                print("\nContent Classification Report:")
                print(classification_report(
                    y_test, 
                    y_pred,
                    target_names=[label_encoder.inverse_transform([i])[0] for i in sorted(set(y_test))]
                ))
                
                # No need for confusion matrix visualization if test set is too small
                if len(set(y_test)) > 1 and len(y_test) >= 5:
                    self.visualize_confusion_matrix(y_test, y_pred, label_encoder)
            except ValueError as e:
                print(f"⚠️ Error with train_test_split: {e}")
                print("⚠️ Falling back to cross-validation")
                
                # Train with cross-validation
                results = self.train_with_cross_validation(X, y)
                
                pipeline = results['model']
                accuracy = results['accuracy']
                
                print(f"📊 Model Accuracy: {accuracy:.2f}")
                print(f"📊 Fold Accuracies: {[f'{acc:.2f}' for acc in results['fold_accuracies']]}")
        
        # Visualize feature importance
        self.visualize_feature_importance(pipeline, X.columns)
        
        # Visualize data distribution if we have enough samples and dimensions
        if len(X) >= 5 and X.shape[1] >= 2:
            self.visualize_data_distribution(X, y, label_encoder)
        
        return {
            'model': pipeline,
            'label_encoder': label_encoder,
            'features': X.columns
        }

    def visualize_confusion_matrix(self, y_true, y_pred, label_encoder):
        """
        Visualize confusion matrix
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            label_encoder (LabelEncoder): Label encoder
        """
        try:
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(y_true, y_pred)
            
            # Get class names
            class_names = label_encoder.inverse_transform(np.unique(np.concatenate([y_true, y_pred])))
            
            # Create heatmap
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names
            )
            
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.savefig('confusion_matrix.png')
            plt.close()
            print("✅ Confusion matrix visualization saved as 'confusion_matrix.png'")
        except Exception as e:
            print(f"⚠️ Could not create confusion matrix: {e}")

    def visualize_feature_importance(self, pipeline, feature_names):
        """
        Visualize feature importance
        
        Args:
            pipeline (Pipeline): Trained pipeline
            feature_names (list): Feature names
        """
        try:
            # Extract feature importance
            classifier = pipeline.named_steps['classifier']
            importances = classifier.feature_importances_
            
            # Sort importance
            indices = np.argsort(importances)[::-1]
            
            # Plot top 10 features or all if less than 10
            n_features = min(10, len(feature_names))
            
            plt.figure(figsize=(12, 8))
            plt.bar(
                range(n_features), 
                importances[indices][:n_features],
                align='center'
            )
            plt.xticks(
                range(n_features), 
                [feature_names[i] for i in indices][:n_features],
                rotation=45,
                ha='right'
            )
            plt.tight_layout()
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.title('Feature Importance')
            plt.savefig('feature_importance.png')
            plt.close()
            print("✅ Feature importance visualization saved as 'feature_importance.png'")
        except Exception as e:
            print(f"⚠️ Could not create feature importance plot: {e}")

    def visualize_data_distribution(self, X, y, label_encoder):
        """
        Visualize data distribution using PCA
        
        Args:
            X (DataFrame): Features
            y (array): Labels
            label_encoder (LabelEncoder): Label encoder
        """
        try:
            # Apply PCA for dimensionality reduction
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            # Create DataFrame for visualization
            pca_df = pd.DataFrame({
                'PCA1': X_pca[:, 0],
                'PCA2': X_pca[:, 1],
                'Grade': label_encoder.inverse_transform(y)
            })
            
            # Create scatter plot
            plt.figure(figsize=(12, 8))
            sns.scatterplot(
                data=pca_df,
                x='PCA1',
                y='PCA2',
                hue='Grade',
                palette='viridis',
                s=100
            )
            plt.title('Data Distribution (PCA)')
            plt.savefig('data_distribution.png')
            plt.close()
            print("✅ Data distribution visualization saved as 'data_distribution.png'")
        except Exception as e:
            print(f"⚠️ Could not create data distribution plot: {e}")
    
    def visualize_readability_distribution(self, texts, grade_levels):
        """
        Visualize readability score distribution by grade level
        
        Args:
            texts (list): Text samples
            grade_levels (list): Grade levels
        """
        try:
            # Calculate readability scores
            readability_scores = [self.get_readability_score(text) for text in texts]
            
            # Create DataFrame for visualization
            df = pd.DataFrame({
                'Grade': grade_levels,
                'Readability': readability_scores
            })
            
            # Get unique grade levels
            unique_grades = df['Grade'].unique()
            
            # Create visualizations based on number of samples
            if len(unique_grades) <= 2:
                # Simple bar chart if only 1-2 grades
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Grade', y='Readability', data=df)
                plt.title('Average Readability Score by Grade Level')
            else:
                # Group by grade and calculate stats
                grade_stats = df.groupby('Grade')['Readability'].agg(['mean', 'std', 'count'])
                
                # Only create boxplot if we have enough data points per grade
                can_use_boxplot = all(grade_stats['count'] >= 2)
                
                if can_use_boxplot:
                    # Box plot for more than 2 grades with enough samples
                    plt.figure(figsize=(12, 8))
                    sns.boxplot(x='Grade', y='Readability', data=df)
                    plt.title('Readability Score Distribution by Grade Level')
                else:
                    # Bar plot with error bars
                    plt.figure(figsize=(12, 8))
                    grade_stats.reset_index(inplace=True)
                    plt.bar(
                        grade_stats['Grade'], 
                        grade_stats['mean'],
                        yerr=grade_stats['std'].fillna(0)
                    )
                    plt.title('Average Readability Score by Grade Level')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('readability_distribution.png')
            plt.close()
            print("✅ Readability distribution visualization saved as 'readability_distribution.png'")
        except Exception as e:
            print(f"⚠️ Could not create readability distribution plot: {e}")

    def recommend_content_level(self, text, trained_model):
        """
        Recommend appropriate content level for a given text
        
        Args:
            text (str): Input text
            trained_model (dict): Trained model dictionary
        
        Returns:
            tuple: Recommended grade level and confidence
        """
        # Extract features
        basic_features = self.extract_features(text)
        
        # Create DataFrame with all expected columns from training
        df = pd.DataFrame([basic_features])
        
        # Make sure all required columns are present
        missing_cols = set(trained_model['features']) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        
        # Remove extra columns not used in training
        extra_cols = set(df.columns) - set(trained_model['features'])
        if extra_cols:
            df = df.drop(columns=extra_cols)
        
        # Ensure columns are in the same order
        df = df[trained_model['features']]
        
        # Predict grade level
        predicted_level_index = trained_model['model'].predict(df)[0]
        recommended_level = trained_model['label_encoder'].inverse_transform([predicted_level_index])[0]
        
        # Get prediction probability
        proba = trained_model['model'].predict_proba(df)[0]
        confidence = proba[predicted_level_index]
        
        return recommended_level, confidence

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
        recommended_level, confidence = self.recommend_content_level(text, trained_model)
        
        # Check if content is appropriate
        is_appropriate = recommended_level in allowed_grades
        
        # Extract features for visualization
        features = self.extract_features(text)
        
        return {
            'original_text': text,
            'recommended_level': recommended_level,
            'confidence': confidence,
            'is_appropriate': is_appropriate,
            'features': features
        }

    def visualize_results(self, results):
        """
        Visualize filtering results
        
        Args:
            results (list): List of filtering results dictionaries
        """
        # Check if we have enough results to visualize
        if len(results) < 2:
            print("⚠️ Not enough results to create meaningful visualizations")
            return
        
        try:
            # Prepare data for visualization
            df = pd.DataFrame([
                {
                    'Text': result['original_text'][:30] + '...' if len(result['original_text']) > 30 else result['original_text'],
                    'Grade': result['recommended_level'],
                    'Confidence': result['confidence'],
                    'Appropriate': result['is_appropriate'],
                    'Readability': result['features']['readability_score'],
                    'Complexity': result['features']['complexity_score'],
                    'Word Count': result['features']['word_count']
                }
                for result in results
            ])
            
            # Create figures directory if it doesn't exist
            import os
            if not os.path.exists('figures'):
                os.makedirs('figures')
            
            # 1. Bar chart of readability scores
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(x='Text', y='Readability', hue='Grade', data=df)
            plt.title('Readability Scores by Text')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('figures/text_readability.png')
            plt.close()
            
            # 2. Scatter plot of readability vs complexity
            plt.figure(figsize=(10, 8))
            sns.scatterplot(
                x='Readability', 
                y='Complexity', 
                hue='Grade',
                size='Word Count',
                sizes=(100, 500),
                data=df
            )
            plt.title('Readability vs Complexity')
            plt.savefig('figures/readability_vs_complexity.png')
            plt.close()
            
            # 3. Bar chart of confidence scores
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(x='Text', y='Confidence', hue='Appropriate', data=df)
            plt.title('Model Confidence by Text')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('figures/confidence_scores.png')
            plt.close()
            
            print("\n✅ Visualizations saved in 'figures' directory:")
            print("  - text_readability.png")
            print("  - readability_vs_complexity.png")
            print("  - confidence_scores.png")
        except Exception as e:
            print(f"⚠️ Could not create result visualizations: {e}")


def main():
    # Expanded training data with more diverse examples and pairs of the same grade level
    training_texts = [
        # Common grade levels with multiple examples
        "The cat sat on the mat.", 
        "The dog ran in the park.",  # Both 2nd Grade
        
        "Mammals are warm-blooded animals that feed their young with milk.",
        "The seasons change because the Earth moves around the sun.",  # Both 3rd Grade
        
        "Photosynthesis is the process by which green plants transform light energy into chemical energy.",
        "The human digestive system breaks down food into nutrients that the body can absorb.",  # Both 6th Grade
        
        "Electricity flows through conductors and powers various electronic devices.",
        "Chemical reactions occur when bonds between atoms are formed or broken.",  # Both 7th Grade
        
        "The mitochondria is known as the powerhouse of the cell, generating ATP through cellular respiration.",
        "The principles of plate tectonics explain the formation of mountains and the occurrence of earthquakes.",  # Both 10th Grade
        
        "Quantum mechanics describes nature at the smallest scales of energy levels of atoms and subatomic particles.",
        "The thermodynamic laws govern energy transfer and transformation processes in physical systems.",  # Both 11th Grade
    ]
    
    training_grades = [
        '2nd Grade', '2nd Grade',
        '3rd Grade', '3rd Grade',
        '6th Grade', '6th Grade',
        '7th Grade', '7th Grade',
        '10th Grade', '10th Grade',
        '11th Grade', '11th Grade'
    ]
    
    # Initialize Content Filtering System
    content_filter = ContentFilteringSystem()
    
    # Visualize readability distribution in training data
    content_filter.visualize_readability_distribution(training_texts, training_grades)
    
    # Train the model
    trained_model = content_filter.train_content_filter_model(
        training_texts, 
        training_grades
    )
    
    # Example texts to filter
    test_texts = [
        "Elephants are large mammals that live in Africa and Asia.",
        "The chemical composition of water is H2O, composed of two hydrogen atoms and one oxygen atom.",
        "Calculus involves the study of continuous change through derivatives and integrals.",
        "The cat jumped over the fence.",
        "The electromagnetic force is one of the four fundamental forces of nature, responsible for electric and magnetic fields.",
        "DNA contains the genetic instructions used in the development and functioning of all known living organisms.",
    ]
    
    # Allowed grade levels for filtering
    allowed_grades = ['3rd Grade', '4th Grade', '5th Grade', '6th Grade']
    
    # Filter and recommend content levels
    results = []
    print("\n🔍 Content Filtering Results:")
    for text in test_texts:
        result = content_filter.filter_content(text, trained_model, allowed_grades)
        results.append(result)
        print("\nText Analysis:")
        print(f"Original Text: {result['original_text']}")
        print(f"Recommended Level: {result['recommended_level']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Appropriate for Grades: {allowed_grades}")
        print(f"Is Appropriate: {result['is_appropriate']}")
        print("Features:")
        for feature, value in result['features'].items():
            print(f"  - {feature}: {value:.2f}" if isinstance(value, float) else f"  - {feature}: {value}")
    
    # Visualize the results
    content_filter.visualize_results(results)

if __name__ == "__main__":
    main()