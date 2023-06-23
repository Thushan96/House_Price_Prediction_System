import streamlit as st
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open('model.pkl', 'rb'))

features_name = ['YearBuilt', 'Area', 'NumBedRooms', 'AreaBedroom', 'BedroomCond', 'NumKitch', 'AreaKitch',
                 'KitchCond', 'Garage', 'GarageArea', 'Electricity', 'AirConditioning', 'NumHearth',
                 'HouseCondition', 'Pool', 'Garden']

# Map categorical values to numerical values
categorical_mapping = {
    'Excellent': 0,
    'Very Good': 1,
    'Good': 2,
    'Average': 3,
    'Typical': 4,
    'Bad': 5,
    'Very Bad': 6,
    'Yes': 1,
    'No': 0,
    'Normal': 0,
    'Abnormal': 1
}

def convert_feature_value(feature, value):
    if isinstance(value, pd.Series):
        return value.apply(lambda x: categorical_mapping.get(x, x))
    return categorical_mapping.get(value, value)

def predict(input_data):
    df = pd.DataFrame([input_data], columns=features_name)
    df = df.apply(lambda x: convert_feature_value(x.name, x))
    house_price = int(model.predict(df.values))
    return house_price


def main():
    st.title('House Price Prediction')
    st.write('Enter the following details to predict the house price:')

    input_data = []
    for feature in features_name:
        if feature.endswith('Cond'):
            options = ['Excellent', 'Very Good', 'Good', 'Average', 'Typical', 'Bad', 'Very Bad']
            input_val = st.selectbox(f'{feature}:', options, index=categorical_mapping['Average'])
        elif feature in ['Garage', 'Electricity', 'Pool', 'Garden']:
            input_val = st.selectbox(f'{feature}:', ['Yes', 'No'])
        elif feature == 'AirConditioning':
            options = ['Excellent', 'Very Good', 'Good', 'Average', 'Typical', 'Bad', 'Very Bad']
            input_val = st.selectbox(f'{feature}:', options, index=categorical_mapping['Excellent'])
        elif feature == 'HouseCondition':
            options = ['Excellent', 'Very Good', 'Good', 'Average', 'Typical', 'Bad', 'Very Bad']
            input_val = st.selectbox(f'{feature}:', options, index=categorical_mapping['Normal'])
        else:
            input_val = st.text_input(f'{feature}:')

        input_data.append(input_val)

    if st.button('Predict'):
        house_price = predict(input_data)
        st.success(f'The Estimated Price for the House is Rs.{house_price}.00')


if __name__ == "__main__":
    main()
