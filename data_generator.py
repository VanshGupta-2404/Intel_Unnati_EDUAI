import pandas as pd
import random
from faker import Faker

fake = Faker('en_IN') 

genders = ["Male", "Female", "Other"]
states = ["Maharashtra", "Karnataka", "Tamil Nadu", "Uttar Pradesh", "West Bengal", "Gujarat", "Rajasthan"]
cities = {
    "Maharashtra": ["Mumbai", "Pune", "Nagpur"],
    "Karnataka": ["Bangalore", "Mysore", "Hubli"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Varanasi"],
    "West Bengal": ["Kolkata", "Asansol", "Siliguri"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur"]
}
parent_occupations = ["Teacher", "Engineer", "Doctor", "Farmer", "Business Owner", "Software Developer", "Government Employee"]
earning_classes = ["Low", "Middle", "High"]
course_names = ["Mathematics", "Science", "English", "History", "Computer Science"]
material_names = ["Textbook", "Video Lesson", "Interactive Quiz", "E-Book"]


num_rows = 10000
data = []

for _ in range(num_rows):
    state = random.choice(states)
    city = random.choice(cities[state])
    
    student = {
        "Name": fake.first_name() + " " + fake.last_name(),
        "Age": random.randint(10, 18),
        "Gender": random.choice(genders),
        "Country": "India",
        "State": state,
        "City": city,
        "Parent Occupation": random.choice(parent_occupations),
        "Earning Class": random.choice(earning_classes),
        "Level of Student": random.randint(1, 12),
        "Level of Course": random.randint(1, 5),
        "Course Name": random.choice(course_names),
        "Assessment Score": random.randint(0, 100),
        "Study Time per Day (hrs)": round(random.uniform(0.5, 5), 1),
        "Material Name": random.choice(material_names),
        "Material Level": random.randint(1, 3),
        "IQ of Student": random.randint(80, 140)
    }
    data.append(student)


df = pd.DataFrame(data)


df.to_csv("indian_student_data.csv", index=False)

print("Dataset created successfully: indian_student_data.csv")