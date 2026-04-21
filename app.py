import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(page_title="Fish Weight Predictor", layout="wide")

st.title("🐟 Fish Weight Prediction App")
st.markdown("""
This app predicts the weight of a fish based on its measurements using a **Linear Regression** model.
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

    # --- SIDEBAR INPUTS ---
    st.sidebar.header("Input Fish Measurements")
    
    selected_species = st.sidebar.selectbox("Select Species", options=le.classes_)
    species_idx = list(le.classes_).index(selected_species)
    
    # Based on your notebook's Lasso/VIF analysis, we use: Species, Vertical_Length, Height
    v_length = st.sidebar.slider("Vertical Length (cm)", 
                                 float(df['Vertical_Length'].min()), 
                                 float(df['Vertical_Length'].max()), 25.0)
    
    height = st.sidebar.slider("Height (cm)", 
                               float(df['Height'].min()), 
                               float(df['Height'].max()), 10.0)

    # --- MODEL TRAINING ---
    # We use the features identified as best in your notebook
    features = ['Species_Encoded', 'Vertical_Length', 'Height']
    X = df[features]
    y = df['Weight']

    model = LinearRegression()
    model.fit(X, y)

    # --- PREDICTION ---
    input_values = np.array([[species_idx, v_length, height]])
    prediction = model.predict(input_values)

    # Display Prediction
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"### Predicted Weight:")
        st.success(f"**{round(max(0, prediction[0]), 2)} Grams**")
    
    with col2:
        st.write("### Input Summary")
        st.write(f"- **Species:** {selected_species}")
        st.write(f"- **Length:** {v_length} cm")
        st.write(f"- **Height:** {height} cm")

    st.divider()

    # --- VISUALIZATION (FIXED HEATMAP) ---
    st.write("### Data Exploration")
    
    show_heatmap = st.checkbox("Show Correlation Heatmap")
    if show_heatmap:
        fig, ax = plt.subplots(figsize=(10, 6))
        # Drop non-numeric for correlation
        numeric_df = df.drop(['Species'], axis=1)
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

    st.write("### Raw Dataset (Preview)")
    st.dataframe(df.head(10))

except FileNotFoundError:
    st.error("Error: 'Fish.csv' not found. Please ensure the dataset is uploaded to your GitHub repository.")
