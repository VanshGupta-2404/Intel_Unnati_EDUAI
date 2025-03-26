## Generate adaptive teaching material dynamically 

import os
import google.generativeai as genai
import PyPDF2
import json
from typing import Dict, List, Optional

class AdaptiveLearningSystem:
    def __init__(self, api_key: str, pdf_path: str):
        """
        Initialize the Adaptive Learning System
        
        Args:
            api_key (str): Google Generative AI API key
            pdf_path (str): Path to the source PDF
        """
        # Configure API
        genai.configure(api_key=api_key)
        
        # Initialize models
        self.text_model = genai.GenerativeModel("gemini-pro")
        
        # Set PDF path
        self.pdf_path = pdf_path
        
        # Create learning profile storage
        self.learning_profiles_path = "learning_profiles.json"
        self.learning_profiles = self.load_learning_profiles()

    def load_learning_profiles(self) -> Dict:
        """
        Load existing learning profiles or create a new file
        
        Returns:
            Dict: Learning profiles dictionary
        """
        try:
            if os.path.exists(self.learning_profiles_path):
                with open(self.learning_profiles_path, 'r') as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            print(f"Error loading learning profiles: {e}")
            return {}

    def save_learning_profiles(self):
        """Save learning profiles to JSON file"""
        try:
            with open(self.learning_profiles_path, 'w') as f:
                json.dump(self.learning_profiles, f, indent=4)
        except Exception as e:
            print(f"Error saving learning profiles: {e}")

    def extract_text_from_pdf(self, page_number: int) -> Optional[str]:
        """
        Extract text from a specific PDF page
        
        Args:
            page_number (int): Page to extract
        
        Returns:
            Optional[str]: Extracted text or None
        """
        try:
            with open(self.pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                
                # Validate page number
                if page_number < 1 or page_number > len(reader.pages):
                    print(f"Invalid page number: {page_number}")
                    return None
                
                page = reader.pages[page_number - 1]
                return page.extract_text()
        
        except FileNotFoundError:
            print(f"PDF file not found: {self.pdf_path}")
            return None
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
            return None

    def generate_adaptive_content(
        self, 
        text: str, 
        student_profile: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Generate adaptive content based on student profile
        
        Args:
            text (str): Source text to adapt
            student_profile (Dict): Student's learning characteristics
        
        Returns:
            Dict: Adapted content for different learning styles
        """
        # Detailed prompts for different learning styles
        learning_styles = {
            "visual": f"""Rewrite the following content with heavy emphasis on visual learning:
            - Use metaphors and visual analogies
            - Include descriptions of potential diagrams or illustrations
            - Break down complex concepts into visual steps
            
            Source Text: {text}""",
            
            "auditory": f"""Adapt the content for auditory learners:
            - Use sound-based metaphors
            - Include potential verbal explanations
            - Create mnemonic devices and rhythmic descriptions
            
            Source Text: {text}""",
            
            "kinesthetic": f"""Transform the content for kinesthetic learners:
            - Describe hands-on activities and experiments
            - Create step-by-step practical applications
            - Use movement-based learning analogies
            
            Source Text: {text}"""
        }
        
        # Generate content for each learning style
        adaptive_content = {}
        for style, prompt in learning_styles.items():
            try:
                response = self.text_model.generate_content(prompt)
                adaptive_content[style] = response.text
            except Exception as e:
                print(f"Error generating {style} content: {e}")
                adaptive_content[style] = "Content generation failed"
        
        return adaptive_content

    def assess_student_learning(
        self, 
        student_id: str, 
        assessment_score: float
    ) -> Dict[str, str]:
        """
        Update student learning profile based on assessment
        
        Args:
            student_id (str): Unique student identifier
            assessment_score (float): Score from recent assessment
        
        Returns:
            Dict: Updated learning recommendations
        """
        # Update or create student profile
        if student_id not in self.learning_profiles:
            self.learning_profiles[student_id] = {
                "total_assessments": 1,
                "average_score": assessment_score,
                "learning_style": self._determine_learning_style(assessment_score)
            }
        else:
            profile = self.learning_profiles[student_id]
            total_assessments = profile["total_assessments"] + 1
            new_avg_score = (
                (profile["average_score"] * profile["total_assessments"]) + assessment_score
            ) / total_assessments
            
            self.learning_profiles[student_id].update({
                "total_assessments": total_assessments,
                "average_score": new_avg_score,
                "learning_style": self._determine_learning_style(new_avg_score)
            })
        
        # Save updated profiles
        self.save_learning_profiles()
        
        return self.learning_profiles[student_id]

    def _determine_learning_style(self, score: float) -> str:
        """
        Determine learning style based on assessment score
        
        Args:
            score (float): Assessment score
        
        Returns:
            str: Recommended learning style
        """
        if score < 50:
            return "kinesthetic"
        elif 50 <= score < 75:
            return "auditory"
        else:
            return "visual"

    def generate_personalized_content(
        self, 
        page_number: int, 
        student_id: str, 
        assessment_score: float
    ) -> Dict[str, str]:
        """
        Generate personalized content for a specific student
        
        Args:
            page_number (int): PDF page to extract content from
            student_id (str): Student identifier
            assessment_score (float): Recent assessment score
        
        Returns:
            Dict: Personalized adaptive content
        """
        # Extract text from PDF
        source_text = self.extract_text_from_pdf(page_number)
        
        if not source_text:
            return {"error": "Could not extract text from PDF"}
        
        # Update student learning profile
        student_profile = self.assess_student_learning(student_id, assessment_score)
        
        # Generate adaptive content
        adaptive_content = self.generate_adaptive_content(
            source_text, 
            student_profile
        )
        
        return {
            "source_text": source_text,
            "student_profile": student_profile,
            "adaptive_content": adaptive_content
        }

def main():
    # Replace with your actual API key and PDF path
    API_KEY = "//API gemini"
    PDF_PATH = "NCERT_BOOK.pdf"
    
    # Initialize Adaptive Learning System
    learning_system = AdaptiveLearningSystem(API_KEY, PDF_PATH)
    
    # Example usage
    page_number = 4
    student_id = "student_001"
    assessment_score = 85  # Example score
    
    personalized_content = learning_system.generate_personalized_content(
        page_number, 
        student_id, 
        assessment_score
    )
    
    # Print results
    print("🔍 Source Text:")
    print(personalized_content.get("source_text", "No source text"))
    
    print("\n📊 Student Profile:")
    print(json.dumps(personalized_content.get("student_profile", {}), indent=2))
    
    print("\n📚 Adaptive Content:")
    for style, content in personalized_content.get("adaptive_content", {}).items():
        print(f"\n{style.upper()} Learning Style:")
        print(content)

if __name__ == "__main__":
    main()