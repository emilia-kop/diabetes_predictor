from flask import Flask, render_template, request
import pickle
from xgboost import XGBClassifier

app = Flask(__name__)

model = XGBClassifier()
model.load_model('diabetes_model.json')

scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    gender_input = request.form['gender']
    age = float(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    smoking_input = request.form['smoking_history']
    bmi = float(request.form['bmi'])
    hba1c = float(request.form['HbA1c_level'])
    glucose = float(request.form['blood_glucose_level'])

    # Binary encode gender exactly as training did: Female=0, Male=1
    gender = 1 if gender_input == 'Male' else 0

    # One-hot encode smoking_history exactly as pd.get_dummies(drop_first=True) did.
    # The dropped baseline category (all zeros) is 'No Info' — it was first alphabetically.
    smoking_current    = 1 if smoking_input == 'current' else 0
    smoking_ever        = 1 if smoking_input == 'ever' else 0
    smoking_former      = 1 if smoking_input == 'former' else 0
    smoking_never       = 1 if smoking_input == 'never' else 0
    smoking_notcurrent  = 1 if smoking_input == 'not current' else 0

    # Scale ONLY the 4 continuous numeric columns, in the same order the scaler was fit on
    numeric_scaled = scaler.transform([[age, bmi, hba1c, glucose]])[0]
    age_s, bmi_s, hba1c_s, glucose_s = numeric_scaled

    # Final column order MUST match X.columns from training.
    # pd.get_dummies() moves the new dummy columns to the END of the
    # dataframe rather than keeping them in smoking_history's original spot,
    # so the real order is:
    # gender, age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level,
    # smoking_history_current, smoking_history_ever, smoking_history_former,
    # smoking_history_never, smoking_history_not current
    features = [[
        gender, age_s, hypertension, heart_disease,
        bmi_s, hba1c_s, glucose_s,
        smoking_current, smoking_ever, smoking_former, smoking_never, smoking_notcurrent
    ]]

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    result = "High Risk of Diabetes" if prediction == 1 else "Low Risk of Diabetes"
    probability_pct = round(probability * 100, 1)

    print(f"Prediction: {result} ({probability_pct}%)")
    return render_template('result.html', prediction=result, probability=probability_pct)

if __name__ == '__main__':
    app.run(port=5501)