#TO CHECK LEVEL OF STUDENT WE WILL INTEGRATE THE TEST

import streamlit as st
import google.generativeai as genai

# Configure Gemini API key
genai.configure(api_key="")  # Replace with your actual API key

# Streamlit UI
st.title("📚 AI-Powered Assessment Generator")

# User Inputs
grade = st.selectbox("Select Grade Level:", ["Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5", "Grade 6", "Grade 7", "Grade 8", "High School"])
subject = st.selectbox("Select Subject:", ["Mathematics", "Science", "English", "History", "Geography", "Physics", "Chemistry", "Biology"])

# Generate Assessment Button
if st.button("Generate Assessment"):
    with st.spinner("Generating your custom assessment..."):
        try:
            # Prompt for AI
            prompt = f"Create a structured assessment for a {grade} level student in {subject}. Include a mix of multiple-choice, short answer, and problem-solving questions."

            # Generate content using Gemini API
            model = genai.GenerativeModel("gemini-1.5-pro-latest")
            response = model.generate_content(prompt)

           
            st.subheader(f"📝 {grade} {subject} Assessment")
            st.markdown(response.text)
        
        except Exception as e:
            st.error(f"Error generating assessment: {e}")


st.markdown("---")
st.markdown("🔹 Powered by **Google Gemini AI** | Designed by **Vansh Gupta**")
