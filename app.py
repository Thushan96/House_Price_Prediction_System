import streamlit as st
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open('model.pkl', 'rb'))

features_name = ['YearBuilt', 'Area', 'NumBedRooms', 'AreaBedroom', 'BedroomCond', 'NumKitch', 'AreaKitch',
                 'KitchCond', 'Garage', 'GarageArea', 'Electricity', 'AirConditioning', 'NumHearth',
                 'HouseCondition', 'Pool', 'Garden']


def predict(input_data):
    df = pd.DataFrame([input_data], columns=features_name)
    house_price = int(model.predict(df))
    return house_price


def main():
    st.title('House Price Prediction')
    st.write('Enter the following details to predict the house price:')

    input_data = []
    for feature in features_name:
        if feature.endswith('Cond'):
            input_val = st.selectbox(f'{feature}:', ['Excellent', 'Very Good', 'Good', 'Average', 'Typical', 'Bad', 'Very Bad'])
        elif feature in ['Garage', 'Electricity', 'Pool', 'Garden']:
            input_val = st.selectbox(f'{feature}:', ['Yes', 'No'])
        else:
            input_val = st.text_input(f'{feature}:')

        input_data.append(input_val)

    if st.button('Predict'):
        house_price = predict(input_data)
        st.success(f'The Estimated Price for the House is Rs.{house_price}.00')


if __name__ == "__main__":
    main()