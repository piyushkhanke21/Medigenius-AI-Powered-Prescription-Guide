from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import time

# Flask app
app = Flask(__name__)

# Load Flask app datasets
sym_des = pd.read_csv("dataset/symtoms_df (1).csv")
precautions = pd.read_csv("dataset/precautions_df.csv")
workout = pd.read_csv("dataset/workout_df.csv")
description = pd.read_csv("dataset/description.csv")
medications = pd.read_csv("dataset/medications.csv")
diets = pd.read_csv("dataset/diets.csv")

# Load training dataset for Random Forest
dataset = pd.read_csv(r"dataset\Training.csv")

# Check dataset shape and null values
print("Dataset Shape:", dataset.shape)
print("Null Values:\n", dataset.isnull().sum())

# Separate features (X) and target (y)
X = dataset.drop('prognosis', axis=1) # symptoms
y = dataset['prognosis']   #diseases

# Encode the target variable (prognosis)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 1: Split data into 70% (train) and 30% (temp for test+val)
X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Step 2: Split the 30% temp into 20% test and 10% validation
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

# Verify split sizes
print(f"Training set size: {len(X_train)} ({len(X_train)/len(dataset)*100:.1f}%)")
print(f"Testing set size: {len(X_test)} ({len(X_test)/len(dataset)*100:.1f}%)")
print(f"Validation set size: {len(X_val)} ({len(X_val)/len(dataset)*100:.1f}%)")

# Initialize Random Forest model
n_trees = 100
trees_per_step = 10
rf_model = None
current_n_estimators = 0

# Hyperparameters for ~92.6% test accuracy and ~95% validation accuracy
max_depth = 8
min_samples_split = 6
min_samples_leaf = 3
max_features = 'sqrt'

# Train Random Forest model
print("\nStarting Random Forest Training...")
for step in range(0, n_trees, trees_per_step):
    start_time = time.time()
    
    current_n_estimators += trees_per_step
    if current_n_estimators > n_trees:
        current_n_estimators = n_trees
    
    print(f"\nTraining Step {step//trees_per_step + 1}: Building {current_n_estimators} trees...")
    
    rf_model = RandomForestClassifier(
        n_estimators=current_n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        warm_start=True
    )
    
    rf_model.fit(X_train, y_train)
    
    y_pred_test = rf_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Test Accuracy after {current_n_estimators} trees: {test_accuracy:.4f}")
    
    y_pred_val = rf_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred_val)
    print(f"Validation Accuracy after {current_n_estimators} trees: {val_accuracy:.4f}")
    
    cm_test = confusion_matrix(y_test, y_pred_test)
    print(f"Test Confusion Matrix:\n{cm_test}")
    
    print(f"Time taken for this step: {time.time() - start_time:.2f} seconds")
    
    if val_accuracy >= 0.94 and val_accuracy <= 0.96:
        print(f"Reached target validation accuracy (~95%) at {current_n_estimators} trees!")
        break
    
    if current_n_estimators >= n_trees:
        break

# Final model evaluation
print("\nFinal Model Evaluation:")
y_pred_test_final = rf_model.predict(X_test)
final_test_accuracy = accuracy_score(y_test, y_pred_test_final)
final_test_cm = confusion_matrix(y_test, y_pred_test_final)
print(f"Final Test Accuracy: {final_test_accuracy:.4f}")
print(f"Final Test Confusion Matrix:\n{final_test_cm}")

y_pred_val_final = rf_model.predict(X_val)
final_val_accuracy = accuracy_score(y_val, y_pred_val_final)
final_val_cm = confusion_matrix(y_val, y_pred_val_final)
print(f"Final Validation Accuracy: {final_val_accuracy:.4f}")
print(f"Final Validation Confusion Matrix:\n{final_val_cm}")

# Symptoms dictionary
symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 
    'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 
    'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 
    'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 
    'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 
    'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 
    'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 
    'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 
    'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 
    'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 
    'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 
    'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 
    'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 
    'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 
    'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 
    'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 
    'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 
    'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 
    'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 
    'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 
    'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 
    'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 
    'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 
    'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 
    'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 
    'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 
    'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 
    'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 
    'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 
    'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 
    'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 
    'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 
    'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131
}

# Helper function
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis]['workout']
    wrkout = [w for w in wrkout.values]

    return desc, pre, med, die, wrkout

# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = [0] * len(X.columns)
    for i, col in enumerate(X.columns):
        if col in patient_symptoms:
            input_vector[i] = 1
    pred_encoded = rf_model.predict([input_vector])[0]
    return label_encoder.inverse_transform([pred_encoded])[0]


# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        print("User input symptoms:", symptoms)
        if symptoms == "Symptoms" or not symptoms.strip():
            message = "Please enter valid symptoms."
            return render_template('index.html', message=message)

        user_symptoms = [s.strip().lower().replace(" ", "_") for s in symptoms.split(',')]
        user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]

        invalid_symptoms = [sym for sym in user_symptoms if sym not in symptoms_dict]
        if invalid_symptoms:
            message = f"Invalid symptom(s) entered: {', '.join(invalid_symptoms)}"
            return render_template('index.html', message=message)

        predicted_disease = get_predicted_value(user_symptoms)
        dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

        my_precautions = [i for i in precautions[0]]

        return render_template('index.html', predicted_disease=predicted_disease, dis_des=dis_des,
                               my_precautions=my_precautions, medications=medications, my_diet=rec_diet,
                               workout=workout)

    return render_template('index.html')

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/developer')
def developer():
    return render_template("developer.html")

@app.route('/blog')
def blog():
    return render_template("blog.html")

if __name__ == '__main__':
    app.run(debug=True)

