import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import randint

# Generate a synthetic dataset
np.random.seed(42)
data = pd.DataFrame({
    'age': np.random.randint(13, 80, 1000),  # Include younger ages
    'weight': np.random.randint(30, 120, 1000),  # Wider range for weight
    'height': np.random.randint(130, 200, 1000),  # Wider range for height
    'blood_pressure': np.random.randint(80, 180, 1000),
    'steps_per_day': np.random.randint(500, 20000, 1000),  # Wider range for steps
    'condition': np.random.choice([0, 1], 1000)  # 0: No condition, 1: Condition
})

# Feature engineering
data['bmi'] = data['weight'] / (data['height'] / 100) ** 2

# Handle missing values
data = data.fillna(data.mean())

# Split data into features and labels
X = data.drop('condition', axis=1)
y = data['condition']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning using RandomizedSearchCV
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 11),
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=100, cv=5, verbose=1, n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

# Train the best model
best_model.fit(X_train, y_train)

def predict_condition(age, weight, height, blood_pressure, steps_per_day):
    # Feature engineering
    bmi = weight / (height / 100) ** 2
    
    # Prepare the input data
    input_data = np.array([[age, weight, height, blood_pressure, steps_per_day, bmi]])
    input_data = scaler.transform(input_data)
    
    # Predict the condition
    prediction = best_model.predict(input_data)
    return "Condition Present" if prediction[0] == 1 else "No Condition"

def recommend_diet(age, weight, height, blood_pressure, condition):
    bmi = weight / (height / 100) ** 2
    diet = "Balanced diet with a mix of proteins, carbohydrates, and fats."

    if condition == "Condition Present":
        diet = "Low-sodium, high-fiber, lean protein, fruits, and vegetables."
        if blood_pressure > 130:
            diet += " Reduce salt intake, avoid processed foods."
        if bmi >= 25:
            diet += " Focus on weight loss: reduce calorie intake, increase physical activity."
        elif bmi < 18.5:
            diet += " Focus on weight gain: increase calorie intake with nutritious foods."
    else:
        if age < 18:
            diet += " Ensure sufficient intake of calcium and vitamin D for bone growth."
        elif age > 60:
            diet += " Ensure sufficient intake of vitamin B12 and fiber to maintain health."
        if steps_per_day < 5000:
            diet += " Increase physical activity to maintain a healthy weight and improve cardiovascular health."
        elif steps_per_day > 15000:
            diet += " Maintain hydration and balance electrolytes due to high physical activity."

    return diet

# Streamlit UI
st.title("AI-Powered Personal Health Care System")

st.header("Enter your details")
age = st.number_input("Age", min_value=1, max_value=120, value=30)
weight = st.number_input("Weight (kg)", min_value=1, max_value=200, value=70)
height = st.number_input("Height (cm)", min_value=50, max_value=250, value=175)
blood_pressure = st.number_input("Blood Pressure", min_value=50, max_value=200, value=120)
steps_per_day = st.number_input("Average Steps per Day", min_value=0, max_value=50000, value=8000)

if st.button("Get Health Prediction and Diet Recommendation"):
    condition_result = predict_condition(age, weight, height, blood_pressure, steps_per_day)
    diet_recommendation = recommend_diet(age, weight, height, blood_pressure, condition_result)
    
    st.subheader("Prediction")
    st.write(condition_result)
    
    st.subheader("Diet Recommendation")
    st.write(diet_recommendation)
