import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display

from preprocess import preprocess_input
from model import model, feature_cols

property_type_options = [
    'Bungalow Property',
    'Converted Flat',
    'Detached Bungalow',
    'Detached House',
    'Detached Property',
    'End Terrace Bungalow',
    'End Terrace House',
    'End Terrace Property',
    'Flat/Maisonette',
    'Mid Terrace Bungalow',
    'Mid Terrace House',
    'Mid Terrace Property',
    'Purpose Built Flat',
    'Semi-Detached Bungalow',
    'Semi-Detached House',
    'Semi-Detached Property',
    'Terrace Property',
    'Terraced',
    'Terraced Bungalow'
    ]

output = widgets.Output()

def predict_price(button):
    with output:
        output.clear_output(wait=True)
        df_input = pd.DataFrame([{
            'postcode': postcode.value,
            'bedrooms': bedrooms.value,
            'bathrooms': bathrooms.value,
            'livingRooms': living_rooms.value,
            'floorAreaSqM': floor_area.value,
            'propertyType': property_type.value,
            'tenure': tenure.value,
            'currentEnergyRating': energy_rating.value
        }])

        try:
            print('Preprocessing inputs...')
            df_processed = preprocess_input(df_input)
            df_processed = df_processed.reindex(columns=feature_cols, fill_value=0)
            print('Preprocessing completed! Predicting...')
            pred = np.exp(model.predict(df_processed))
            print(f'Predicted Property Price: £{pred[0]:,.0f}')
        except ValueError as error_msg:
            print(f'Error: {error_msg}')

postcode = widgets.Text(value='SW3 5BL', description='Postcode:', placeholder='e.g. SW3 5BL')
bedrooms = widgets.IntSlider(value=1, min=1, max=6, step=1, description='Bedrooms:')
bathrooms = widgets.IntSlider(value=1,min=1, max=4, step=1, description='Bathrooms:')
living_rooms = widgets.IntSlider(value=1, min=0, max=3, step=1, description='Living Rooms:')
floor_area = widgets.IntSlider(value=37, min=20, max=250, step=1, description='Floor Area (in sqm):')
property_type = widgets.Dropdown(value='Bungalow Property', options=property_type_options, description='Property Type:')
tenure = widgets.Dropdown(value='Freehold', options=['Leasehold', 'Freehold'], description='Tenure:')
energy_rating = widgets.Dropdown(value='E', options=['A', 'B', 'C', 'D', 'E', 'F', 'G'], description='Energy Rating:')

predict_button = widgets.Button(description='Predict')
predict_button._click_handlers.callbacks.clear()
predict_button.on_click(predict_price)

display(postcode, bedrooms, bathrooms, living_rooms, floor_area, property_type, tenure, energy_rating, predict_button, output)