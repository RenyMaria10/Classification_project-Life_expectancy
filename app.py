import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration for a premium look
st.set_page_config(
    page_title="Country Status Classification",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern, vibrant, premium design
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1e1e2f 0%, #252542 100%);
        color: #f8f9fa;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    .main-header {
        text-align: center;
        background: -webkit-linear-gradient(45deg, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        margin-bottom: 0.5rem;
        animation: fadeInDown 1s ease-out;
    }
    
    .sub-header {
        text-align: center;
        color: #a0aabf !important;
        font-size: 1.2rem;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    
    /* Glassmorphism containers */
    .glass-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 2rem;
    }
    
    .glass-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4);
    }
    
    /* Inputs */
    .stNumberInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px;
    }
    .stNumberInput > div > div > input:focus {
        border-color: #4facfe !important;
        box-shadow: 0 0 0 1px #4facfe !important;
    }
    
    .stNumberInput label {
        color: #e0e6ed !important;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        color: white;
        border-radius: 50px;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        width: 100%;
        transition: all 0.4s ease;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.6);
    }
    
    /* Prediction Result Card */
    .result-card {
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.2) 0%, rgba(0, 242, 254, 0.2) 100%);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(79, 172, 254, 0.4);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        animation: fadeIn 0.8s ease-out;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    .result-title {
        color: #a0aabf;
        font-size: 1.2rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 1rem;
    }
    
    .result-value {
        font-size: 4rem;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown('<h1 class="main-header">Country Status Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Determine if a country is Developed or Developing based on key socio-economic factors.</p>', unsafe_allow_html=True)

# Load the trained model and calculate the scaler dynamically
@st.cache_resource
def load_model_and_scaler():
    from sklearn.preprocessing import StandardScaler
    try:
        model = joblib.load('knn_model.pkl')
        df = pd.read_csv('Life Expectancy Data.csv')
        
        FEATURES = [
            'Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure', 
            'Hepatitis B', 'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 
            'Total expenditure', 'Diphtheria ', ' HIV/AIDS', 'GDP', 
            ' thinness  1-19 years', ' thinness 5-9 years', 
            'Income composition of resources', 'Schooling'
        ]
        
        # The model requires scaled features. Fit a StandardScaler to the original data.
        X_all = df[FEATURES].fillna(df[FEATURES].median())
        scaler = StandardScaler()
        scaler.fit(X_all)
        
        # Calculate medians for default values
        medians = X_all.median().to_dict()
        
        return model, scaler, FEATURES, medians
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        return None, None, None, None

model, scaler, FEATURES, medians = load_model_and_scaler()

if model is None:
    st.stop()

# Layout
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown('<h3>Input Key Metrics</h3>', unsafe_allow_html=True)
    
    with st.form("classification_form"):
        # We only surface the most important features to avoid overwhelming the user
        input_data = medians.copy()
        
        c1, c2 = st.columns(2)
        
        input_data['GDP'] = c1.number_input("GDP per capita (USD)", 
                                           value=float(medians['GDP']), 
                                           format="%.2f",
                                           help="Gross Domestic Product per capita")
                                           
        input_data['Income composition of resources'] = c2.number_input("Income Composition", 
                                           value=float(medians['Income composition of resources']), 
                                           format="%.3f",
                                           help="Human Development Index in terms of income composition of resources (0.0 - 1.0)")
        
        input_data['Schooling'] = c1.number_input("Years of Schooling", 
                                           value=float(medians['Schooling']), 
                                           format="%.1f",
                                           help="Number of years of Schooling")
                                           
        input_data[' BMI '] = c2.number_input("Average BMI", 
                                           value=float(medians[' BMI ']), 
                                           format="%.1f",
                                           help="Average Body Mass Index of entire population")
        
        input_data['Adult Mortality'] = c1.number_input("Adult Mortality Rate", 
                                           value=float(medians['Adult Mortality']), 
                                           format="%.1f",
                                           help="Probability of dying between 15 and 60 years (per 1000 population)")
                                           
        input_data[' HIV/AIDS'] = c2.number_input("HIV/AIDS Deaths", 
                                           value=float(medians[' HIV/AIDS']), 
                                           format="%.2f",
                                           help="Deaths per 1000 live births HIV/AIDS (0-4 years)")
        
        st.markdown("<br>", unsafe_allow_html=True)
        submit_button = st.form_submit_button("Classify Country Status")
        
    st.markdown('</div>', unsafe_allow_html=True)

if submit_button:
    # Convert input data to dataframe and strictly enforce the column order
    input_df = pd.DataFrame([input_data])[FEATURES]
    
    try:
        # Scale the input data
        X_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        
        # Map integer prediction to string label
        # 0 corresponds to Developing, 1 corresponds to Developed
        status_label = "Developed" if prediction == 1 else "Developing"
        
        # Determine styling based on prediction
        if status_label == 'Developed':
            icon = "🏙️"
            color_gradient = "linear-gradient(135deg, #00b09b 0%, #96c93d 100%)"
        else:
            icon = "🌱"
            color_gradient = "linear-gradient(135deg, #f6d365 0%, #fda085 100%)"
            
        with col2:
            st.markdown(f"""
            <div class="result-card" style="background: {color_gradient}20; border-color: {color_gradient.split(',')[1].strip()};">
                <div class="result-title">Classification Result</div>
                <div style="font-size: 5rem; margin-bottom: 1rem;">{icon}</div>
                <h2 class="result-value" style="background: {color_gradient}; -webkit-background-clip: text;">{status_label}</h2>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Prediction failed: {e}")
