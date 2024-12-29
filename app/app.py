import os
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from PIL import Image

# Load the saved model# Define the models directory
models_dir = 'app/models'

# Function to load a model
def load_model(model_name):
    model_path = os.path.join(models_dir, f'{model_name}_model.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    else:
        st.error(f"Model file for '{model_name}' not found in {models_dir}")
        return None

# Main function for the Streamlit app
def main():
    st.title('Iris Flower Classification')
    st.sidebar.write('This app classifies Iris flowers into different species based on the provided features.')

    # Create inputs in the sidebar for the user to enter feature values
    st.sidebar.header('Input Features')
    sepal_length = st.sidebar.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, step=0.1)
    sepal_width = st.sidebar.number_input('Sepal Width (cm)', min_value=0.0, max_value=10.0, step=0.1)
    petal_length = st.sidebar.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, step=0.1)
    petal_width = st.sidebar.number_input('Petal Width (cm)', min_value=0.0, max_value=10.0, step=0.1)

    # Prepare the input data as a DataFrame
    input_data = pd.DataFrame({
        'Sepal Length': [sepal_length],
        'Sepal Width': [sepal_width],
        'Petal Length': [petal_length],
        'Petal Width': [petal_width]
    })

    # Load models
    model_name = st.sidebar.selectbox('Select Classifier', ['LDA', 'KNN', 'DT', 'NB', 'SVC', 'RF'])
    model = load_model(model_name)

    # Load images for each flower species
    setosa = Image.open('setosa.png')
    versicolor = Image.open('versicolor.png')
    virginica = Image.open('virginica.png')

    # Define a dictionary for species names and images
    species_dict = {
        'Iris-setosa': {'name': 'Setosa', 'image': setosa},
        'Iris-versicolor': {'name': 'Versicolor', 'image': versicolor},
        'Iris-virginica': {'name': 'Virginica', 'image': virginica}
    }

    # Predict when the user clicks the "Classify" button
    if st.sidebar.button('Classify'):
        prediction = model.predict(input_data)
        flower_species = prediction[0]
        species_info = species_dict.get(flower_species, {'name': 'Unknown species', 'image': None})
        
        # Display the predicted species name and image
        st.write(f'The predicted species is: {species_info["name"]}')
        if species_info['image']:
            st.image(species_info['image'], caption=species_info["name"], use_container_width=True)

if __name__ == '__main__':
    main()
