import pandas as pd 
import numpy as np 
import joblib


import streamlit as st
from catboost import CatBoostRegressor

loaded_model = CatBoostRegressor()
loaded_model.load_model('./Final_codes/catboost_model.cbm')
brand_encoder = joblib.load('./Final_codes/brand_encoder.pkl')

st.header('Car Price Prediction Web Site (FORZA FERRARI)')

df = pd.read_csv('./Final_codes/exported_used_cars.csv')






brands = sorted(df['brand'].unique())
selected_brand = st.selectbox("Select a car brand:", ["Select a brand"] + brands)

# Márka kiválasztása után betöltődnek a modellek
if selected_brand != "Select a brand":
    # Az adott márkához tartozó egyedi modellek
    models = sorted(df[df['brand'] == selected_brand]['base_model'].unique())
    model = st.selectbox(f"Select a model for {selected_brand}:", ["Select a model"] + models)

    # Kiválasztott modell megjelenítése
    if model != "Select a model":
        st.write(f"You selected brand: **{selected_brand}** and model: **{model}**")
else:
    st.write("Please select a brand first.")






fuel_types = ['Gasoline', 'Hybrid', 'E85 Flex Fuel', 'Diesel', 'Plug-In Hybrid', 'LPG', 'Electric', 'Hydrogen']
selected_fuel_type = st.selectbox("Select fuel type:", fuel_types)

age = st.slider("Model Year (pl. 2020):",  2000,2024)
milage = st.slider("Mileage (pl. 50000):",  0,200000)
#engine = st.text_input("Engine (pl. 2.0L):", "")


hp = st.text_input('HP:', '')
displacement = st.text_input('Engine Displacement (L), 0 if electic or hydrogen:', '')

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



ext_col_unique = sorted(df['ext_col'].unique())
ext_col = st.selectbox("Select an exterior colour:", ext_col_unique)
int_col_unique = sorted(df['int_col'].unique())
int_col = st.selectbox("Select an interior colour:", int_col_unique)



accident_options = ["None reported", "At least 1 accident or damage reported"]
selected_accident = st.selectbox("Choose an accident history:", accident_options)

transmission = ['Automatic', 'Manual', 'CVT']
selected_transmission = st.selectbox('Choose a gear type:', transmission)

if valid_input:
    if st.button("Predict"):
        input_data_model = pd.DataFrame(
        [[selected_brand, milage, selected_fuel_type, selected_transmission, ext_col, int_col, selected_accident, age, hp, displacement, model]],
        columns=['brand', 'milage','fuel_type','transmission', 'ext_col','int_col','accident', 'Age', 'Horsepower','Engine_Displacement','base_model'])
        
        input_data_model['Age'] = 2024 - input_data_model['Age']

        input_data_model['base_model'] = input_data_model['base_model']
        input_data_model['ext_col'] = input_data_model['ext_col']
        input_data_model['int_col'] = input_data_model['int_col']

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
        input_data_model['transmission'] = input_data_model['transmission'].replace({'Automatic':1,'Manual':2, 'Other':3})
        input_data_model['accident'] = input_data_model['accident'].replace({'At least 1 accident or damage reported':1,'None reported':0})

        input_data_model['clean_accident_interaction'] = input_data_model['accident'] + input_data_model['accident']

        brand_data = pd.DataFrame(input_data_model['brand'], columns=['brand'])
        brand_transformed = brand_encoder.transform(brand_data)

        # A végső adathalmaz előkészítése
        input_data_model = pd.concat([brand_transformed, input_data_model.drop(columns=['brand'])], axis=1)

        # A modell bemenetéhez igazított oszlopsorrend
        expected_columns = ['brand_0', 'brand_1', 'brand_2', 'brand_3', 'brand_4', 'brand_5', 'milage', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'accident', 'Age', 'Horsepower', 'Engine_Displacement', 'base_model', 'clean_accident_interaction']
        input_data_model = input_data_model.reindex(columns=expected_columns, fill_value=0)





        car_price_log = loaded_model.predict(input_data_model)

        car_price_original = round(10 ** car_price_log[0])

        st.markdown('Car Price is going to be $'+ str(car_price_original))
