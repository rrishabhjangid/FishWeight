import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# 1. Page Configuration
st.set_page_config(
    page_title="Fish Weight Predictor",
    page_icon="🐟",
    layout="wide"
)

# 2. Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stNumberInput, .stSelectbox {
        border-radius: 10px;
    }
    .prediction-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #007bff;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Data Loading
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Fish.csv')
        df.rename(columns={'Length1': 'Vertical_Length', 'Length3': 'Cross_Length'}, inplace=True)
        le = LabelEncoder()
        df['Species_Encoded'] = le.fit_transform(df['Species'])
        return df, le
    except:
        return None, None

df, le = load_data()

if df is not None:
    # --- HEADER ---
    st.title("⚖️ Fish Weight Estimator")
    st.write("Provide the species and physical dimensions below to estimate the weight.")

    # --- INPUT SECTION ---
    with st.container():
        st.subheader("Physical Characteristics")
        
        # Using columns to organize inputs better
        row1_col1, row1_col2 = st.columns(2)
        
        with row1_col1:
            selected_species = st.selectbox(
                "🐟 Species", 
                options=le.classes_,
                help="Select the specific breed of the fish."
            )
            species_idx = list(le.classes_).index(selected_species)

        with row1_col2:
            # We use the median of the dataset as the default value for a 'better' starting point
            v_length = st.number_input(
                "📏 Vertical Length (cm)", 
                min_value=0.0, 
                value=float(df['Vertical_Length'].median()),
                format="%.2f",
                help="Measurement from the nose to the beginning of the tail."
            )

        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            height = st.number_input(
                "📐 Body Height (cm)", 
                min_value=0.0, 
                value=float(df['Height'].median()),
                format="%.2f",
                help="Maximum vertical thickness of the fish body."
            )
        
        with row2_col2:
            st.write(" ") # Spacer
            st.write(" ") # Spacer
            predict_btn = st.button("Calculate Weight", use_container_width=True)

    # --- MODEL & PREDICTION ---
    # Training on relevant features identified in your notebook
    X = df[['Species_Encoded', 'Vertical_Length', 'Height']]
    y = df['Weight']
    model = LinearRegression().fit(X, y)

    if predict_btn:
        input_data = np.array([[species_idx, v_length, height]])
        prediction = model.predict(input_data)
        result = round(max(0, prediction[0]), 2)

        st.markdown("---")
        
        # Displaying result in a custom formatted card
        st.markdown(f"""
            <div class="prediction-card">
                <h3 style='margin-top:0;'>Estimation Results</h3>
                <p>Based on a <b>Linear Regression</b> model trained on 159 fish samples:</p>
                <h1 style='color: #007bff;'>{result} Grams</h1>
            </div>
            """, unsafe_allow_html=True)
        
        if result > 0:
            st.balloons()
else:
    st.error("Missing Dataset: Please upload 'Fish.csv' to your GitHub repository.")
