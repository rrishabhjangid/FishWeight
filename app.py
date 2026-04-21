import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.title("Fish Weight Prediction App")

# Load and process data (Simplified based on your notebook)
@st.cache_data
def load_data():
    df = pd.read_csv('Fish.csv')
    df.rename(columns={'Length1': 'Vertical_Length', 'Length3': 'Cross_Length'}, inplace=True)
    le = LabelEncoder()
    df['Species'] = le.fit_transform(df['Species'])
    # Keeping features identified in your VIF/Lasso analysis: Species, Vertical_Length, Height
    return df, le

df, le = load_data()

# Sidebar for inputs
st.sidebar.header("Input Fish Measurements")
species_choice = st.sidebar.selectbox("Species", le.classes_)
v_length = st.sidebar.number_input("Vertical Length (cm)", value=25.0)
height = st.sidebar.number_input("Height (cm)", value=10.0)

# Model Training (Doing it on the fly for simplicity, or load a pickle file)
X = df[['Species', 'Vertical_Length', 'Height']]
y = df['Weight']

model = LinearRegression()
model.fit(X, y)

# Prediction
input_data = np.array([[le.transform([species_choice])[0], v_length, height]])
prediction = model.predict(input_data)

st.write(f"### Predicted Weight of {species_choice}:")
st.success(f"{round(prediction[0], 2)} Grams")

st.write("Correlation Heatmap of the dataset:")
st.image("https://raw.githubusercontent.com/streamlit/docs/main/public/images/tutorials/pandas-tutorial/dataframe.png") # Placeholder
