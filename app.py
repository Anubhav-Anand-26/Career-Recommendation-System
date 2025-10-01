import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

data = pd.read_csv('student-scores.csv')

# Preprocess the data
data.drop(columns=['id', 'first_name', 'last_name', 'email'], inplace=True)

# Calculate total_score and average_score
data['total_score'] = (data["math_score"] + data["history_score"] + data["physics_score"] +
                       data["chemistry_score"] + data["biology_score"] + data["english_score"] + 
                       data["geography_score"])
data['average_score'] = data['total_score'] / 7

# Map categorical values
gender_map = {'male': 0, 'female': 1}
part_time_job_map = {False: 0, True: 1}
extracurricular_activities_map = {False: 0, True: 1}
career_aspiration_map = {
    'Lawyer': 0, 'Doctor': 1, 'Government Officer': 2, 'Artist': 3, 'Unknown': 4,
    'Software Engineer': 5, 'Teacher': 6, 'Business Owner': 7, 'Scientist': 8,
    'Banker': 9, 'Writer': 10, 'Accountant': 11, 'Designer': 12,
    'Construction Engineer': 13, 'Game Developer': 14, 'Stock Investor': 15,
    'Real Estate Developer': 16
}

# Apply mappings
data['gender'] = data['gender'].map(gender_map)
data['part_time_job'] = data['part_time_job'].map(part_time_job_map)
data['extracurricular_activities'] = data['extracurricular_activities'].map(extracurricular_activities_map)
data['career_aspiration'] = data['career_aspiration'].map(career_aspiration_map)

# Separate features and target variable
X = data.drop('career_aspiration', axis=1)
y = data['career_aspiration']

# Balance the dataset with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Accuracy metrics (optional)
y_pred = model.predict(X_test_scaled)
st.write("Model Accuracy:", accuracy_score(y_test, y_pred))

# Class names for career aspirations
class_names = ['Lawyer', 'Doctor', 'Government Officer', 'Artist', 'Unknown',
               'Software Engineer', 'Teacher', 'Business Owner', 'Scientist',
               'Banker', 'Writer', 'Accountant', 'Designer',
               'Construction Engineer', 'Game Developer', 'Stock Investor',
               'Real Estate Developer']

# Define recommendation function
def Recommendations(gender, part_time_job, absence_days, extracurricular_activities,
                    weekly_self_study_hours, math_score, history_score, physics_score,
                    chemistry_score, biology_score, english_score, geography_score,
                    total_score, average_score):
    
    # Encode categorical features
    gender_encoded = 1 if gender.lower() == 'female' else 0
    part_time_job_encoded = 1 if part_time_job else 0
    extracurricular_activities_encoded = 1 if extracurricular_activities else 0
    
    # Create feature array
    feature_array = np.array([[gender_encoded, part_time_job_encoded, absence_days, extracurricular_activities_encoded,
                               weekly_self_study_hours, math_score, history_score, physics_score,
                               chemistry_score, biology_score, english_score, geography_score,
                               total_score, average_score]])
    
    # Scale features
    scaled_features = scaler.transform(feature_array)
    
    # Predict probabilities
    probabilities = model.predict_proba(scaled_features)
    
    # Get top 5 recommendations
    top_classes_idx = np.argsort(-probabilities[0])[:5]
    top_classes_names_probs = [(class_names[idx], probabilities[0][idx]) for idx in top_classes_idx]
    
    return top_classes_names_probs

# Streamlit app UI
st.title("Career Recommendation System")

# Input fields for the app
gender = st.selectbox("Gender", ["Male", "Female"])
part_time_job = st.checkbox("Part-time job")
absence_days = st.number_input("Absence days", min_value=0, step=1)
extracurricular_activities = st.checkbox("Extracurricular activities")
weekly_self_study_hours = st.number_input("Weekly self-study hours", min_value=0, step=1)
math_score = st.number_input("Math score", min_value=0, max_value=100, step=1)
history_score = st.number_input("History score", min_value=0, max_value=100, step=1)
physics_score = st.number_input("Physics score", min_value=0, max_value=100, step=1)
chemistry_score = st.number_input("Chemistry score", min_value=0, max_value=100, step=1)
biology_score = st.number_input("Biology score", min_value=0, max_value=100, step=1)
english_score = st.number_input("English score", min_value=0, max_value=100, step=1)
geography_score = st.number_input("Geography score", min_value=0, max_value=100, step=1)

# Calculate total_score and average_score based on user input
total_score = (math_score + history_score + physics_score + chemistry_score + 
               biology_score + english_score + geography_score)
average_score = total_score / 7

# Get recommendations when button is clicked
if st.button("Get Career Recommendations"):
    recommendations = Recommendations(gender, part_time_job, absence_days, extracurricular_activities,
                                      weekly_self_study_hours, math_score, history_score, physics_score,
                                      chemistry_score, biology_score, english_score, geography_score,
                                      total_score, average_score)

    st.write("Top recommended careers with probabilities:")
    for class_names, probabilities in recommendations:
        st.write(f"{class_names}: {probabilities:.2f}")
