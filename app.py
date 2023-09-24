from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('logistic_model.pkl')
scaler = joblib.load('scaler.pkl')
EXPECTED_COLUMNS = joblib.load('training_columns.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        data = {
            # Extract data from home.html
            'Marital status' : [int(request.form['Estado civil'])],
            'Application mode': [int(request.form['Aplication mode'])],
            'Application order' : [float(request.form['Aplication order'])],
            'Course' : [int(request.form['Course'])],
            "Daytime/evening attendance\t" : [int(request.form['Daytime'])],
            'Previous qualification' : [int(request.form['Previous qualification'])],
            'Previous qualification (grade)' : [float(request.form['Previous qualification grade'])],
            'Nacionality' : [int(request.form['Nacionality'])],
            "Mother's qualification" : [int(request.form['Mother qualification'])],
            "Father's qualification" : [int(request.form['Father qualification'])],
            "Mother's occupation" : [int(request.form['Mother occupation'])],
            "Father's occupation" : [int(request.form['Father occupation'])],
            'Admission grade' : [float(request.form['Admision grade'])],
            'Displaced' : [int(request.form['Displaced'])],
            'Educational special needs' : [int(request.form['Special needs'])],
            'Debtor' : [int(request.form['Deptor'])],
            'Tuition fees up to date' : [int(request.form['Tuition fees'])],
            'Gender' : [int(request.form['Gender'])],
            'Scholarship holder' : [int(request.form['Scholarship'])],
            'Age at enrollment' : [float(request.form['Age'])],
            'International' : [int(request.form['International'])],
            'Curricular units 1st sem (credited)' : [float(request.form['Curricular credited'])],
            'Curricular units 1st sem (enrolled)' : [float(request.form['Curricular enrolled'])],
            'Curricular units 1st sem (evaluations)' : [float(request.form['Curricular evaluations'])],
            'Curricular units 1st sem (approved)' : [float(request.form['Curricular approved'])],
            'Curricular units 1st sem (grade)' : [float(request.form['Curricular grade'])],
            'Curricular units 1st sem (without evaluations)' : [float(request.form['Curricular without evaluations'])],
            'Curricular units 2nd sem (credited)' : [float(request.form['Curricular credited2'])],
            'Curricular units 2nd sem (enrolled)' : [float(request.form['Curricular enrolled2'])],
            'Curricular units 2nd sem (evaluations)' : [float(request.form['Curricular evaluations2'])],
            'Curricular units 2nd sem (approved' : [float(request.form['Curricular approved2'])],
            'Curricular units 2nd sem (grade)' : [float(request.form['Curricular grade2'])],
            'Curricular units 2nd sem (without evaluations)' : [float(request.form['Curricular without evaluations2'])],
            'Unemployment rate' : [float(request.form['Unemployment rate'])],
            'Inflation rate' : [float(request.form['Inflation rate'])],
            'GDP' : [float(request.form['GDP'])]
        }
        
        # Create a dataframe using 'data' dictionary
        df = pd.DataFrame(data)
        
        # One - hot encode the new data
        df_encoded = pd.get_dummies(df)
        
        # Check for missing columns and add them if they are missing
        for col in EXPECTED_COLUMNS:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        #Reorder columns based on the expected order
        df_encoded = df_encoded[EXPECTED_COLUMNS]

        #Scale the features
        features_scaled = scaler.transform(df_encoded)
        features_scaled = pd.DataFrame(features_scaled, columns=EXPECTED_COLUMNS)

        # Make prediction
        prediction_val = model.predict(features_scaled)
        if prediction_val == 1:
            prediction = 'Dropout'
        else:
            prediction = 'Graduate'
    return render_template('home.html', prediction=prediction)
if __name__ == '__main__':
    app.run(debug=True)




