from turtle import width
import pandas as pd
import streamlit as st
from joblib import load
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
import lime
import lime.lime_tabular

CURRENT_THEME = "light"
df = pd.read_csv("heart.csv")
X = df.drop('target', axis = 1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
im = Image.open("clipart-223147.jpg")

def calculate():
     with st.spinner('Calculating...'):
          st.write("Probability of heart disease: {}%".format(probability))
          st.write(fig)
          explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_train) ,class_names=['not disease', 'disease'], 
                                                   feature_names = list(df.columns),
#                                                    categorical_features=categorical_features, 
#                                                    categorical_names=categorical_names, 
#                                                    kernel_width=3, verbose=False
                                                    discretize_continuous=True
                                                  )
          exp = explainer.explain_instance(np.asarray([age, sex, cp, trestbps, chol, fbs, restecg, 
                                    thalach, exang, oldpeak, slope, ca, thal]), model.predict_proba, 
                                 num_features=13, top_labels=1)
          st.components.v1.html(exp.as_html(show_table=False, show_all=True, show_predicted_value = False, predict_proba=False), height=450)

st.set_page_config(page_title= 'Heart disease probability', page_icon = im,layout = 'wide')
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
     ca = st.selectbox('number of major vessels (0-3) colored by flourosopy (ca)',(0, 1, 2, 3))

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
    mode = "gauge",
    gauge = {'shape': "bullet", 'axis':{'range':[0,100]}, 'bar':{'color':color}},
    value = probability,
    delta = {'reference': 100},
    domain = {'x': [0, 1], 'y': [0, 1]},
    ))
fig.update_layout(height = 200,
                    width = 600)
with col4:
     if _claculate:
        calculate()     