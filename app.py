import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

app=Flask(__name__)

model = pickle.load(open('random_forest_model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))
encoder= pickle.load(open('encoder.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    pain_nonangnal_pain=0
    pain_typical_angina=0
    rest_normal=0
    slope_upsloping=0
    if request.method=='POST':
        age=int(request.form['age'])
        resting_blood_pressure=int(request.form['resting_blood_pressure'])
        cholesterol=int(request.form['cholesterol'])
        fasting_blood_sugar=int(request.form['fasting_blood_sugar'])
        max_heart_rate=int(request.form['max_heart_rate'])
        exercise_angina=int(request.form['exercise_angina'])
        st_depression=float(request.form['st_depression'])
        s_male=request.form['s_male']
        if(s_male=='male'):
            s_male=1
        else:
            s_male=0
        pain_atypical_angina=request.form['pain_atypical_angina']
        if(pain_atypical_angina=='atypical_angina'):
            pain_atypical_angina=1
            pain_nonangnal_pain=0
            pain_typical_angina=0
        elif(pain_atypical_angina=='nonangnal_pain'):
            pain_atypical_angina=0
            pain_nonangnal_pain=1
            pain_typical_angina=0
        elif(pain_atypical_angina=='typical_angina'):
            pain_atypical_angina=0
            pain_nonangnal_pain=0
            pain_typical_angina=1
        else:
            pain_atypical_angina=0
            pain_nonangnal_pain=0
            pain_typical_angina=0
        rest_left_ventricular_hypertrophy=request.form['rest_left_ventricular_hypertrophy']
        if(rest_left_ventricular_hypertrophy=='left_ventricular_hypertrophy'):
            rest_left_ventricular_hypertrophy=1
            rest_normal=0
        elif(rest_left_ventricular_hypertrophy=='normal'):
            rest_left_ventricular_hypertrophy=0
            rest_normal=0
        else:
            rest_left_ventricular_hypertrophy=0
            rest_normal=0
        slope_flat=request.form['slope_flat']
        if(slope_flat=='flat'):
            slope_flat=1
            slope_upsloping=0
        elif(slope_flat=='upsloping'):
            slope_flat=0
            slope_upsloping=1
        else:
            slope_flat=0
            slope_upsloping=0
    #features=[float(x) for x in request.form.values()]
    features=np.array([[age,resting_blood_pressure,cholesterol,fasting_blood_sugar,max_heart_rate,exercise_angina,st_depression,s_male,pain_atypical_angina,rest_left_ventricular_hypertrophy,slope_flat]])
    final_features=[np.array(features)]
    final_features=scaler.transform(final_features)
    final_features=encoder.transform(final_features)
    prediction=model.predict(final_features)
    print('final features',final_features)
    print('prediction:',prediction)
    output=round(prediction[0],2)
    print(output)

    if output==0:
        return render_template('index.html', prediction_text='THE PATIENT IS NOT LIKELY TO HAVE A HEART FAILURE')
    else:
        return render_template('index.html', prediction_text='THE PATIENT IS LIKELY TO HAVE A HEART FAILURE')

@app.route('/predict_api', methods=['POST'])
def results():
    data=request.get_json(force=True)
    prediction=model.predict([np.array(list(data.values()))])

    output=prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)