import pandas as pd
import streamlit as st
from joblib import load
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
import shap


CURRENT_THEME = "light"
df = pd.read_csv("heart.csv")
X = df.drop('target', axis = 1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
im = Image.open("clipart-223147.jpg")

st.set_page_config(page_title= 'Heart disease probability', page_icon = im,layout = 'wide')



def calculate():
     with st.spinner('Calculating...'):
          st.write("### Probability[*](https://github.com/huseyn-guliyev/heart-disease-app) of heart disease: {}%".format(probability))
          st.write(fig)
          explainer = shap.TreeExplainer(model)

          # Calculate Shap values
          shap_values = explainer.shap_values(X)
          explainer = shap.TreeExplainer(model)

          # Calculate Shap values
          shap_values = explainer.shap_values(np.asarray([age, sex, cp, trestbps, chol, fbs, restecg, 
                                    thalach, exang, oldpeak, slope, ca, thal]).reshape(1,13))
          shap.initjs()
          fig2 = shap.force_plot(explainer.expected_value[1], shap_values[1], 
                         np.asarray([age, sex, cp, trestbps, chol, fbs, restecg, 
                                    thalach, exang, oldpeak, slope, ca, thal]).reshape(1,13), 
                         feature_names = X.columns, link="logit")
          shap_html = f"<head>{shap.getjs()}</head><body>{fig2.html()}</body>"
          st.write('Explanation')
          st.components.v1.html(shap_html)
          st.write('''
          Note that 
          \n
          "sex" will show 1 if "sex" is "male", otherwise 0 ,
          \n
          "fbs" will show 1 if "fbs" is "True", otherwise 0,
          \n
          "exang" will show 1 if "exang" is "yes", otherwise 0
          ''')

col1, col2, col3, col4 = st.columns([1, 1, 1, 3])


with col1:
     #1 age
     age = int(st.text_input('age (age)', '25'))

     #2 sex
     _sex = st.selectbox(
               'Select gender (sex)',
               ('male', 'female'))

     sex = 1 if _sex == 'male' else 0

     #3 cp
     cp = st.selectbox(
               'Chest pain type (cp)',
               (0, 1,2,3))

     #4
     trestbps = int(st.text_input('resting blood pressure (trestbps)', '125'))

     #5
     # serum cholestoral in mg/dl
     
     chol = int(st.text_input('serum cholestoral in mg/dl (chol)', '257'))
with col2:
     #6
     _fbs = st.selectbox('fasting blood sugar > 120 mg/dl (fbs)',('True', 'False'))

     fbs = 1 if _fbs == 'True' else 0

     #7
     # resting electrocardiographic results
     restecg = st.selectbox('resting electrocardiographic results (restecg)',(0, 1, 2))
     #8
     # maximum heart rate achieved

with col2:
     thalach = int(st.text_input('maximum heart rate achieved (thalach)', '162'))

     #9 
     # exercise induced angina 
     _exang = st.selectbox('exercise induced angina (exang)',('yes', 'no'))

     exang = 1 if _exang == 'yes' else 0
with col3:

     #10
     # ST depression induced by exercise relative to rest 
     oldpeak = float(st.text_input('ST depression induced by exercise relative to rest (oldpeak)', '0'))

     #11
     # the slope of the peak exercise ST segment 
     slope = st.selectbox('the slope of the peak exercise ST segment (slope)',(0, 1, 2))

     #12
     # number of major vessels (0-3) colored by flourosopy 
     ca = st.selectbox('number of major vessels (0-3) colored by flourosopy (ca)',(0, 1, 2, 3, 4))

     #13
     # thal: 0 = normal; 1 = fixed defect; 2 = reversable defect
     thal = st.selectbox('0 = normal; 1 = fixed defect; 2 = reversable defect (thal)',(0, 1, 2, 3))
     _claculate = st.button("Show Results")
          # calculate()

model = load('lightgbm.joblib')
probability = round(model.predict_proba(np.asarray([age, sex, cp, trestbps, chol, fbs, restecg, 
                                    thalach, exang, oldpeak, slope, ca, thal]).reshape(1,13))[0][1]*100, 2)


color = 'red' if probability > 50 else 'green'

fig = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = probability,
    mode = "gauge+number",
    gauge = {'axis': {'range': [None, 100]},
		'bar': {'color': color},
             'steps' : [
                 {'range': [0, 50], 'color': "rgb(204,255,197)"},
                 {'range': [50, 100], 'color': "rgb(255,185,185)"}],
             'threshold' : {'line': {'color': "rgb(0,0,255)", 'width': 4}, 'thickness': 0.75, 'value': 50}}))

fig.update_layout(height = 350,
                    width = 600)

with col4:
     if _claculate:
          calculate()
