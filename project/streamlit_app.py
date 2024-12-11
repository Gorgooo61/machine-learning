import pandas as pd 
import numpy as np 
import pickle as pk 
import streamlit as st
import joblib

learning_model = pk.load(open('car_price_model.pkl','rb'))
brand_encoder = joblib.load('brand_encoder.pkl')
fuel_type_encoder = joblib.load('fuel_type_encoder.pkl')

st.header('Car Price Prediction Web Site (FORZA FERRARI)')

df = pd.read_csv('used_cars.csv')


brands = sorted(df['brand'].unique())
selected_brand = st.selectbox("Choose a car brand:", brands)

model = st.text_input("Model:", "")

fuel_types = ['Gasoline', 'Hybrid', 'E85 Flex Fuel', 'Diesel', 'Plug-In Hybrid', 'LPG', 'Electric', 'Hydrogen']
selected_fuel_type = st.selectbox("Select fuel type:", fuel_types)

age = st.slider("Model Year (pl. 2020):",  1994,2024)
milage = st.slider("Milage (pl. 50000):",  0,250000)
#engine = st.text_input("Engine (pl. 2.0L):", "")


hp = st.text_input('HP:', '')
displacement = st.text_input('Engine Displacement (L) 0 if electic or hydrogen:', '')

try:
    if hp.strip() != '' and displacement.strip() != '':
        hp = int(hp)
        displacement = float(displacement)
        valid_input = True
    else:
        valid_input = False
except ValueError:
    st.error("Please only enter numbers in the HP and Engine Displacement fields!")
    valid_input = False


ext_col = st.text_input("Exterior color (ext_col):", "")
int_col = st.text_input("Interior color (int_col):", "")

accident_options = ["None reported", "At least 1 accident or damage reported"]
selected_accident = st.selectbox("Choose an accident history:", accident_options)

transmission = ['Automatic', 'Manual', 'CVT']
selected_transmission = st.selectbox('Choose a gear type:', transmission)

clean_title_options = ["Yes", "No"]
selected_clean_title = st.selectbox("Is it clean?:", clean_title_options)

if valid_input:
    if st.button("Predict"):
        input_data_model = pd.DataFrame(
        [[selected_brand, milage, selected_fuel_type, selected_transmission, selected_accident, selected_clean_title, age, hp, displacement, model, int_col, ext_col]],
        columns=['brand','milage','fuel_type','transmission','accident','clean_title', 'Age', 'Horsepower','Engine_Displacement','model','int_col','ext_col'])
        
        input_data_model['Age'] = 2024 - input_data_model['Age']
        input_data_model=input_data_model.drop(['model'], axis=1)

        def AutomaticOrManual(transmission):
            if 'Automatic' in transmission:
                return 'Automatic'
            elif 'Manual' in transmission:
                return 'Manual'
            else :
                return 'Other'
        input_data_model['transmission'] = input_data_model['transmission'].apply(AutomaticOrManual)

        input_data_model['Horsepower'] = input_data_model['Horsepower'].astype(float)
        input_data_model['Engine_Displacement'] = input_data_model['Engine_Displacement'].astype(float)
        input_data_model=input_data_model.drop(['ext_col', 'int_col'], axis=1)
        input_data_model['transmission'] = input_data_model['transmission'].replace({'Automatic':1,'Manual':2, 'Other':3})
        input_data_model['accident'] = input_data_model['accident'].replace({'At least 1 accident or damage reported':1,'None reported':0})
        input_data_model['clean_title'] = input_data_model['clean_title'].replace({'Yes':1,'No':0})


        input_data_model['brand'] = brand_encoder.transform(input_data_model['brand'])
        input_data_model['fuel_type'] = fuel_type_encoder.transform(input_data_model['fuel_type'])



        car_price_log = learning_model.predict(input_data_model)

        car_price_original = round(10 ** car_price_log[0])

        st.markdown('Car Price is going to be $'+ str(car_price_original))