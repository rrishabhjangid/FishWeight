import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(page_title="Fish Weight Predictor", layout="centered")

st.title("🐟 Fish Weight Prediction App")
st.markdown("""
Enter the measurements of the fish below to predict its weight using our trained **Linear Regression** model.
""")

# Load and process data
@st.cache_data
def load_and_clean_data():
    # Ensure Fish.csv is in the same GitHub folder
    df = pd.read_csv('Fish.csv')
    
    # Rename columns as done in your notebook
    df.rename(columns={
        'Length1': 'Vertical_Length',
        'Length2': 'Diagonal_Length',
        'Length3': 'Cross_Length'
    }, inplace=True)
    
    # Encode Species
    le = LabelEncoder()
    df['Species_Encoded'] = le.fit_transform(df['Species'])
    
    return df, le

try:
    df, le = load_and_clean_data()

    # --- INPUT SECTION ---
    st.write("### 1. Select Species and Dimensions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_species = st.selectbox("Fish Species", options=le.classes_)
        species_idx = list(le.classes_).index(selected_species)

    with col2:
        # Use ranges based on the actual dataset
        v_length = st.number_input("Vertical Length (cm)", 
                                    min_value=float(df['Vertical_Length'].min()), 
                                    max_value=float(df['Vertical_Length'].max()), 
                                    value=25.0)
        
        height = st.number_input("Height (cm)", 
                                  min_value=float(df['Height'].min()), 
                                  max_value=float(df['Height'].max()), 
                                  value=10.0)

    # --- MODEL TRAINING ---
    # We use the features identified as best in your notebook: Species, Vertical Length, and Height
    features = ['Species_Encoded', 'Vertical_Length', 'Height']
    X = df[features]
    y = df['Weight']

    model = LinearRegression()
    model.fit(X, y)

    # --- PREDICTION ---
    st.divider()
    if st.button("Predict Weight"):
        input_values = np.array([[species_idx, v_length, height]])
        prediction = model.predict(input_values)
        
        # Ensure we don't show negative weights
        final_weight = round(max(0, prediction[0]), 2)
        
        st.write("### 2. Prediction Result")
        st.success(f"The estimated weight for this **{selected_species}** is **{final_weight} Grams**.")
        
        # Additional context for the user
        st.info(f"Model Inputs: Species Code {species_idx}, Length {v_length}cm, Height {height}cm")

except FileNotFoundError:
    st.error("Error: 'Fish.csv' not found. Please ensure the dataset is uploaded to your GitHub repository.")
