import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# 1. Set Page Configuration for a cleaner look
st.set_page_config(
    page_title="Fish Weight Predictor",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS to enhance UI styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div.stButton > button:first-child {
        background-color: #007bff;
        color: white;
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Load and process data
@st.cache_data
def load_and_clean_data():
    try:
        df = pd.read_csv('Fish.csv')
        # Standardizing names as per your notebook analysis
        df.rename(columns={'Length1': 'Vertical_Length', 'Length3': 'Cross_Length'}, inplace=True)
        le = LabelEncoder()
        df['Species_Encoded'] = le.fit_transform(df['Species'])
        return df, le
    except FileNotFoundError:
        return None, None

df, le = load_and_clean_data()

if df is not None:
    # --- SIDEBAR: Configuration ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2829/2829818.png", width=100)
        st.title("Settings")
        st.info("This model uses Linear Regression to estimate fish weight based on Species, Vertical Length, and Height.")
        st.divider()
        st.write("### Data Range Info")
        st.caption(f"Min Weight: {df['Weight'].min()}g")
        st.caption(f"Max Weight: {df['Weight'].max()}g")

    # --- MAIN UI: Header ---
    st.title("⚖️ Fish Weight Prediction Tool")
    st.markdown("---")

    # --- INPUT SECTION: Organized into Columns ---
    st.subheader("1. Enter Fish Specifications")
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        selected_species = st.selectbox(
            "Select Fish Species", 
            options=le.classes_,
            help="Choose the species of the fish you want to weigh."
        )
        species_idx = list(le.classes_).index(selected_species)

    with col2:
        v_length = st.number_input(
            "Vertical Length (cm)", 
            min_value=0.0, 
            max_value=100.0, 
            value=float(df['Vertical_Length'].median()),
            step=0.1
        )
        height = st.number_input(
            "Fish Height (cm)", 
            min_value=0.0, 
            max_value=50.0, 
            value=float(df['Height'].median()),
            step=0.1
        )

    # --- MODEL PROCESSING ---
    features = ['Species_Encoded', 'Vertical_Length', 'Height']
    X = df[features]
    y = df['Weight']

    model = LinearRegression()
    model.fit(X, y)

    # --- PREDICTION AND RESULTS ---
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Calculate Estimated Weight"):
        input_values = np.array([[species_idx, v_length, height]])
        prediction = model.predict(input_values)
        
        # Clip negative results to 0
        final_weight = round(max(0, prediction[0]), 2)
        
        st.divider()
        st.subheader("2. Resulting Prediction")
        
        # Displaying result in a metric-like card
        res_col1, res_col2, res_col3 = st.columns([1, 2, 1])
        with res_col2:
            st.metric(label=f"Estimated Weight for {selected_species}", value=f"{final_weight} Grams")
            
            if final_weight > 1000:
                st.warning("That's a big fish! 🐋")
            elif final_weight > 0:
                st.success("Weight calculated successfully! ✅")
            else:
                st.error("Invalid measurements; weight cannot be predicted.")

else:
    st.error("⚠️ Dataset not found! Please upload 'Fish.csv' to your GitHub repository.")
