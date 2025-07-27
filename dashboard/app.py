import sys
import os

# Get the directory of the current script (app.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (which is the project root 'federated learning')
project_root = os.path.join(current_dir, '..')
# Add the project root to sys.path
sys.path.append(project_root)

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Page Configuration
st.set_page_config(
    page_title="MedAI - Federated Diabetes Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS Styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        padding: 0rem 1rem;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        background: linear-gradient(45deg, #fff, #e0e7ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        font-weight: 400;
        opacity: 0.9;
        max-width: 600px;
        margin: 0 auto;
    }
    
    /* Login Card */
    .login-container {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 60vh;
        margin: -1rem;
        padding: 2rem;
    }
    
    .login-card {
        background: white;
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 25px 50px rgba(0,0,0,0.15);
        width: 100%;
        max-width: 400px;
        text-align: center;
    }
    
    .login-title {
        font-size: 2rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    
    .login-subtitle {
        color: #6b7280;
        margin-bottom: 2rem;
        font-size: 1rem;
    }
    
    /* Modern Cards */
    .modern-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #f1f5f9;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .modern-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    /* Info Cards */
    .info-card {
        background: linear-gradient(135deg, #fef7ff 0%, #f3e8ff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #8b5cf6;
        margin: 1rem 0;
    }
    
    .info-card h4 {
        color: #6b21a8;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .info-card p {
        color: #5b21b6;
        margin: 0;
        line-height: 1.6;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #7dd3fc;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0284c7;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        color: #0369a1;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        font-size: 1rem;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }
    
    /* Form Styling */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid #e5e7eb;
        transition: border-color 0.3s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .stTextInput > div > div {
        border-radius: 8px;
        border: 2px solid #e5e7eb;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Progress Indicators */
    .progress-indicator {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        height: 8px;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
    }
    
    .status-success {
        background: #dcfce7;
        color: #166534;
    }
    
    .status-warning {
        background: #fef3c7;
        color: #92400e;
    }
    
    .status-error {
        background: #fee2e2;
        color: #991b1b;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .hero-subtitle {
            font-size: 1.1rem;
        }
        
        .login-card {
            margin: 1rem;
            padding: 2rem;
        }
        
        .modern-card {
            padding: 1.5rem;
        }
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #5a67d8 0%, #6b46c1 100%);
    }
</style>
""", unsafe_allow_html=True)

def show_login_page():

    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('''
            <div style="text-align: center; margin-bottom: 2rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🩺</div>
                <h1 class="login-title">NeuroNode Login</h1>
                <p class="login-subtitle">Secure access to federated diabetes prediction</p>
        ''', unsafe_allow_html=True) 

        
        with st.form("login_form", clear_on_submit=False):
            st.markdown("#### Sign In")
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            
            col_login, col_info = st.columns([1, 1])
            
            with col_login:
                submitted = st.form_submit_button("🚀 Sign In", use_container_width=True)
            
            with col_info:
                with st.expander("ℹ️ Demo Credentials"):
                    st.write("**Username:** admin")
                    st.write("**Password:** admin123")
            
            if submitted:
                if username == "admin" and password == "admin123":
                    st.session_state.logged_in = True
                    st.success("✅ Login successful! Redirecting...")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Please try again.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_hero_section():
    """Display hero section with key information"""
    st.markdown('''
    <div class="hero-section fade-in">
        <h1 class="hero-title">🩺 MedAI Dashboard</h1>
        <p class="hero-subtitle">
            Advanced federated learning system for diabetes risk prediction. 
            Our AI analyzes health patterns across multiple sources while maintaining privacy.
        </p>
    </div>
    ''', unsafe_allow_html=True)

def create_info_cards():
    """Create informational cards explaining the system"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('''
        <div class="info-card">
            <h4>🔒 What is Federated Learning?</h4>
            <p>A privacy-preserving approach where AI models learn from data across multiple locations without sharing sensitive information.</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown('''
        <div class="info-card">
            <h4>🎯 How It Works</h4>
            <p>Multiple healthcare providers train local models on their data, then share only model updates to create a global prediction system.</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown('''
        <div class="info-card">
            <h4>📊 Benefits</h4>
            <p>Better predictions through diverse data sources while keeping patient information secure and private at each location.</p>
        </div>
        ''', unsafe_allow_html=True)

def create_health_form():
    """Enhanced health prediction form"""
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    
    # Form explanations
    st.markdown("""
    ### 🧪 Health Assessment Tool
    
    Please fill out this comprehensive health questionnaire. Our AI will analyze your responses to assess diabetes risk.
    
    ⚠️ **Important:** This is for educational purposes only and should not replace professional medical advice.
    """)
    
    # Health mappings with better descriptions
    yes_no = {"No": 0.0, "Yes": 1.0}
    gender_map = {"Male": 1.0, "Female": 0.0}
    gen_health_map = {
        "Excellent": 1.0, 
        "Very Good": 2.0, 
        "Good": 3.0, 
        "Fair": 4.0, 
        "Poor": 5.0
    }
    
    age_map_brfss = {
        "18-24": 1.0, "25-29": 2.0, "30-34": 3.0, "35-39": 4.0,
        "40-44": 5.0, "45-49": 6.0, "50-54": 7.0, "55-59": 8.0,
        "60-64": 9.0, "65-69": 10.0, "70-74": 11.0, "75-79": 12.0,
        "80+": 13.0
    }

    education_map = {
        "Never Attended School": 1.0, "Elementary": 2.0, "Middle School": 3.0,
        "High School": 4.0, "Some College": 5.0, "Associate Degree": 6.0,
        "Bachelor's Degree": 7.0, "Graduate Degree": 8.0
    }
    
    income_map = {
        "<$10K": 1.0, "$10K–$15K": 2.0, "$15K–$20K": 3.0,
        "$20K–$25K": 4.0, "$25K–$35K": 5.0, "$35K–$50K": 6.0,
        "$50K–$75K": 7.0, ">$75K": 8.0
    }

    with st.form("enhanced_predict_form"):
        # Personal Information Section
        st.subheader("👤 Personal Information")
        col1, col2 = st.columns(2)
        
        with col1:
            sex = st.selectbox("Gender", list(gender_map.keys()), help="Biological sex")
            age = st.selectbox("Age Range", list(age_map_brfss.keys()), index=len(age_map_brfss)//2, help="Select your age range")
        
        with col2:
            edu = st.selectbox("Education Level", list(education_map.keys()), index=3, help="Highest level of education completed")
            income = st.selectbox("Annual Income", list(income_map.keys()), index=len(income_map)//2, help="Household income range")

        st.divider()
        
        # Health Conditions Section
        st.subheader("🏥 Medical History")
        col1, col2 = st.columns(2)
        
        with col1:
            highbp = st.radio("High Blood Pressure", list(yes_no.keys()), help="Have you been diagnosed with high blood pressure?")
            highchol = st.radio("High Cholesterol", list(yes_no.keys()), help="Have you been diagnosed with high cholesterol?")
            stroke = st.radio("History of Stroke", list(yes_no.keys()), help="Have you ever had a stroke?")
            heart = st.radio("Heart Disease", list(yes_no.keys()), help="Have you ever been diagnosed with heart disease?")
        
        with col2:
            cholcheck = st.radio("Cholesterol Check (Last 5 Years)", list(yes_no.keys()), index=1, help="Have you had your cholesterol checked in the past 5 years?")
            healthcare = st.radio("Health Insurance Coverage", list(yes_no.keys()), index=1, help="Do you have any kind of health care coverage?")
            nodoc = st.radio("Avoided Doctor Due to Cost", list(yes_no.keys()), help="Was there a time when you needed to see a doctor but could not because of cost?")
            walk = st.radio("Difficulty Walking/Climbing Stairs", list(yes_no.keys()), help="Do you have serious difficulty walking or climbing stairs?")

        st.divider()
        
        # Lifestyle Section
        st.subheader("🏃‍♂️ Lifestyle Factors")
        col1, col2 = st.columns(2)
        
        with col1:
            bmi = st.slider("Body Mass Index (BMI)", 10.0, 50.0, 25.0, 0.1, 
                          help="BMI = weight(kg) / height(m)². Normal range: 18.5-24.9")
            smoker = st.radio("Smoking History", list(yes_no.keys()), help="Have you smoked at least 100 cigarettes in your entire life?")
            alcohol = st.radio("Heavy Alcohol Use", list(yes_no.keys()), help="Heavy drinking (men: ≥14 drinks/week, women: ≥7 drinks/week)")
        
        with col2:
            phys = st.radio("Physical Activity (Last 30 Days)", list(yes_no.keys()), index=1, help="Physical activity outside of regular job in past 30 days")
            fruits = st.radio("Daily Fruit Consumption", list(yes_no.keys()), index=1, help="Do you consume fruit 1 or more times per day?")
            veggies = st.radio("Daily Vegetable Consumption", list(yes_no.keys()), index=1, help="Do you consume vegetables 1 or more times per day?")

        st.divider()
        
        # Health Status Section
        st.subheader("💪 Current Health Status")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            genhlth = st.selectbox("General Health", list(gen_health_map.keys()), index=2, help="How would you rate your general health?")
        
        with col2:
            ment = st.slider("Poor Mental Health Days (Last 30)", 0.0, 30.0, 0.0, 1.0, help="Days in past 30 when mental health was not good")
        
        with col3:
            physhlth = st.slider("Poor Physical Health Days (Last 30)", 0.0, 30.0, 0.0, 1.0, help="Days in past 30 when physical health was not good")

        st.divider()
        
        # Submit button with enhanced styling
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button("🚀 Analyze My Diabetes Risk", use_container_width=True)

        if submitted:
            # Create prediction data
            single_feature_dict = {
                "HighBP": yes_no[highbp],
                "HighChol": yes_no[highchol],
                "CholCheck": yes_no[cholcheck],
                "BMI": bmi,
                "Smoker": yes_no[smoker],
                "Stroke": yes_no[stroke],
                "HeartDiseaseorAttack": yes_no[heart],
                "PhysActivity": yes_no[phys],
                "Fruits": yes_no[fruits],
                "Veggies": yes_no[veggies],
                "HvyAlcoholConsump": yes_no[alcohol],
                "AnyHealthcare": yes_no[healthcare],
                "NoDocbcCost": yes_no[nodoc],
                "GenHlth": gen_health_map[genhlth],
                "MentHlth": ment,
                "PhysHlth": physhlth,
                "DiffWalk": yes_no[walk],
                "Sex": gender_map[sex],
                "Age": age_map_brfss[age],
                "Education": education_map[edu],
                "Income": income_map[income],
            }

            input_data_for_api = {"features": [single_feature_dict]}

            try:
                with st.spinner("🔍 Analyzing your health data with our AI model..."):
                    response = requests.post("http://localhost:8000/", json=input_data_for_api, timeout=30)
                    
                    if response.status_code == 200:
                        prediction_result = response.json()
                        
                        # Enhanced result display
                        st.markdown("---")
                        st.markdown("### 📋 Risk Assessment Results")
                        
                        if prediction_result["prediction"] == 1:
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); 
                                        padding: 2rem; border-radius: 12px; border-left: 5px solid #ef4444; margin: 1rem 0;">
                                <h3 style="color: #dc2626; margin-bottom: 1rem;">⚠️ Higher Risk Detected</h3>
                                <p style="color: #991b1b; font-size: 1.1rem; margin-bottom: 1rem;">
                                    Based on the provided information, our AI model indicates you may be at higher risk for diabetes.
                                </p>
                                <p style="color: #7f1d1d; font-size: 0.95rem;">
                                    <strong>Important:</strong> This is a screening tool only. Please consult with a healthcare professional 
                                    for proper diagnosis and personalized medical advice.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); 
                                        padding: 2rem; border-radius: 12px; border-left: 5px solid #22c55e; margin: 1rem 0;">
                                <h3 style="color: #16a34a; margin-bottom: 1rem;">✅ Lower Risk Indicated</h3>
                                <p style="color: #15803d; font-size: 1.1rem; margin-bottom: 1rem;">
                                    Based on the provided information, our AI model indicates you may be at lower risk for diabetes.
                                </p>
                                <p style="color: #166534; font-size: 0.95rem;">
                                    <strong>Keep it up!</strong> Continue maintaining healthy lifestyle habits. 
                                    Regular check-ups with your healthcare provider are still recommended.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Add recommendations
                        with st.expander("💡 General Health Recommendations", expanded=True):
                            st.markdown("""
                            **Healthy Lifestyle Tips:**
                            - 🥗 Maintain a balanced diet rich in vegetables and fruits
                            - 🏃‍♂️ Engage in regular physical activity (150 minutes/week moderate exercise)
                            - ⚖️ Maintain a healthy weight (BMI 18.5-24.9)
                            - 🚭 Avoid tobacco and limit alcohol consumption
                            - 🩺 Regular health screenings and check-ups
                            - 😌 Manage stress and prioritize mental health
                            """)
                            
                    else:
                        st.error(f"❌ Server Error ({response.status_code}): {response.text}")
                        
            except requests.exceptions.ConnectionError:
                st.error("🚨 Cannot connect to the prediction service. Please ensure the backend server is running.")
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. Please try again.")
            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

@st.cache_data(ttl=300)
def fetch_metrics():
    """Enhanced metrics fetching with better error handling"""
    try:
        response = requests.get("http://localhost:8000/metrics/", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        global_df = pd.DataFrame(data.get("global_metrics", []))
        client_df = pd.DataFrame(data.get("client_metrics", [])) 

        if not global_df.empty:
            global_df["iteration"] = global_df["iteration"].astype(int)
            global_df = global_df.sort_values("iteration").set_index("iteration")
        if not client_df.empty:
            client_df["iteration"] = client_df["iteration"].astype(int)
            client_df["client_id"] = client_df["client_id"].astype(int)
            client_df["model_name"] = client_df["model_name"].astype(str)
            client_df = client_df.sort_values(["client_id", "model_name", "iteration"])
        return global_df, client_df
    except Exception as e:
        st.error(f"⚠️ Could not fetch metrics: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def create_plotly_chart(df, metric, chart_type="line", title=""):
    """Create enhanced Plotly charts"""
    if df.empty:
        return None
    
    if chart_type == "line":
        fig = px.line(
            df.reset_index(), 
            x="iteration", 
            y=metric,
            title=title,
            markers=True,
            line_shape="spline"
        )
        fig.update_traces(
            line=dict(color="#667eea", width=3),
            marker=dict(size=8, color="#764ba2")
        )
    else:
        fig = px.bar(
            df.reset_index(), 
            x="iteration", 
            y=metric,
            title=title,
            color_discrete_sequence=["#667eea"]
        )
    
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=12),
        title_font=dict(size=16, color="#1f2937"),
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)"),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def main():
    """Main application logic"""
    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # Show login page if not logged in
    if not st.session_state.logged_in:
        show_login_page()
        return

    # Sidebar with enhanced styling
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <div style="font-size: 2rem; margin-bottom: 1rem;">🩺</div>
            <h2 style="color: white; margin-bottom: 0;">MedAI</h2>
            <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">Federated Learning Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("**👤 Session Info**")
        st.info("Logged in as: **Administrator**")
        
        if st.button("🔓 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("**📊 System Status**")
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                st.success("🟢 Backend Online")
            else:
                st.warning("🟡 Backend Issues")
        except:
            st.error("🔴 Backend Offline")

    # Main content
    show_hero_section()
    create_info_cards()

    # Enhanced tabs with better icons and descriptions
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Health Check", 
        "📊 AI Model Progress", 
        "👥 Healthcare Partners", 
        "📈 Model Comparison", 
        "🌍 System Overview"
    ])

    with tab1:
        st.markdown("### 🔍 Personal Health Risk Assessment")
        st.markdown("Use our advanced federated AI to assess your diabetes risk based on comprehensive health factors.")
        create_health_form()

    # Fetch metrics for other tabs
    global_metrics_df, client_metrics_df = fetch_metrics()

    with tab2:
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Global AI Model Performance")
        st.markdown("""
        This shows how our federated AI model improves over time as it learns from multiple healthcare partners 
        without accessing individual patient data.
        """)

        if global_metrics_df.empty:
            st.markdown("""
            <div class="info-card">
                <h4>🔄 Model Training in Progress</h4>
                <p>The federated learning system is currently training. Metrics will appear once training rounds are completed.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Metrics selection
            metric_options = [col for col in global_metrics_df.columns if col not in ['iteration']]
            
            col1, col2 = st.columns([2, 1])
            with col1:
                selected_metric = st.selectbox(
                    "📈 Choose Performance Metric:",
                    metric_options,
                    index=metric_options.index("accuracy") if "accuracy" in metric_options else 0,
                    help="Select which aspect of model performance to visualize"
                )
            
            with col2:
                chart_type = st.selectbox(
                    "📊 Chart Type:",
                    ["Line Chart", "Bar Chart"],
                    help="Choose visualization style"
                )

            if selected_metric:
                # Create enhanced chart
                chart_type_key = "line" if chart_type == "Line Chart" else "bar"
                title = f"Global Model {selected_metric.replace('_', ' ').title()} Over Training Rounds"
                
                fig = create_plotly_chart(global_metrics_df, selected_metric, chart_type_key, title)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show latest metrics in cards
                st.markdown("#### 📋 Latest Performance Metrics")
                latest_data = global_metrics_df.iloc[-1]
                
                cols = st.columns(len(metric_options))
                for i, metric in enumerate(metric_options):
                    with cols[i]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{latest_data[metric]:.3f}</div>
                            <div class="metric-label">{metric.replace('_', ' ').title()}</div>
                        </div>
                        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        st.markdown("### 👥 Healthcare Partner Performance")
        st.markdown("""
        Each healthcare partner trains their own AI model locally, then contributes to the global model. 
        This preserves patient privacy while improving overall prediction accuracy.
        """)

        if client_metrics_df.empty:
            st.markdown("""
            <div class="info-card">
                <h4>🏥 Waiting for Healthcare Partners</h4>
                <p>Healthcare partner metrics will appear once they begin participating in the federated learning process.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            unique_clients = sorted(client_metrics_df["client_id"].unique())

            for client_id in unique_clients:
                with st.expander(f"🏥 Healthcare Partner #{client_id} Performance", expanded=False):
                    client_data = client_metrics_df[client_metrics_df["client_id"] == client_id].copy()
                    
                    if client_data.empty:
                        st.info(f"No data available for Healthcare Partner #{client_id}.")
                        continue
                    
                    numeric_cols = [col for col in client_data.columns if col not in ["client_id", "iteration", "model_name", "fit_status"]]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        selected_metric = st.selectbox(
                            "📊 Performance Metric:",
                            numeric_cols,
                            index=numeric_cols.index("accuracy") if "accuracy" in numeric_cols else 0,
                            key=f"partner_{client_id}_metric"
                        )
                    
                    with col2:
                        unique_models = sorted(client_data["model_name"].unique())
                        selected_model = st.selectbox(
                            "🤖 AI Model:",
                            unique_models,
                            key=f"partner_{client_id}_model"
                        )
                    
                    with col3:
                        chart_style = st.selectbox(
                            "📈 Chart Style:",
                            ["Line Chart", "Bar Chart"],
                            key=f"partner_{client_id}_chart"
                        )

                    if selected_metric and selected_model:
                        filtered_data = client_data[client_data["model_name"] == selected_model].copy()
                        filtered_data.set_index("iteration", inplace=True)
                        
                        if not filtered_data.empty:
                            chart_type_key = "line" if chart_style == "Line Chart" else "bar"
                            title = f"Partner #{client_id} - {selected_model} Model Performance"
                            
                            fig = create_plotly_chart(filtered_data, selected_metric, chart_type_key, title)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Training status information
                            st.markdown("**🔄 Training Status:**")
                            status_data = filtered_data[['fit_status']].reset_index()
                            for _, row in status_data.iterrows():
                                status = row['fit_status']
                                iteration = row['iteration']
                                status_class = "status-success" if status == "completed" else "status-warning"
                                st.markdown(f'<span class="status-badge {status_class}">Round {iteration}: {status}</span>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        st.markdown("### 📈 AI Model Comparison")
        st.markdown("""
        Compare different AI models from the same healthcare partner to see which approach works best 
        for diabetes prediction in their specific patient population.
        """)

        if client_metrics_df.empty:
            st.markdown("""
            <div class="info-card">
                <h4>🔬 Model Comparison Unavailable</h4>
                <p>Model comparison data will be available once healthcare partners have trained multiple model types.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            unique_clients_comparison = sorted(client_metrics_df["client_id"].unique())
            
            selected_client = st.selectbox(
                "🏥 Select Healthcare Partner:",
                options=unique_clients_comparison,
                key="comparison_client_select",
                help="Choose which healthcare partner's models to compare"
            )

            if selected_client:
                client_comparison_data = client_metrics_df[
                    client_metrics_df["client_id"] == selected_client
                ].copy()

                if not client_comparison_data.empty:
                    available_models = sorted(client_comparison_data["model_name"].unique())
                    
                    if len(available_models) > 1:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            selected_models = st.multiselect(
                                "🤖 Select AI Models to Compare:",
                                options=available_models,
                                default=available_models[:2],
                                key="models_to_compare"
                            )
                        
                        with col2:
                            comparison_metric = st.selectbox(
                                "📊 Comparison Metric:",
                                [col for col in client_comparison_data.columns if col not in ["client_id", "iteration", "model_name", "fit_status"]],
                                index=0
                            )

                        if selected_models and len(selected_models) > 1:
                            # Create comparison chart
                            fig = go.Figure()
                            colors = px.colors.qualitative.Set1
                            
                            for i, model in enumerate(selected_models):
                                model_data = client_comparison_data[
                                    client_comparison_data["model_name"] == model
                                ].sort_values("iteration")
                                
                                fig.add_trace(go.Scatter(
                                    x=model_data["iteration"],
                                    y=model_data[comparison_metric],
                                    mode='lines+markers',
                                    name=f"{model}",
                                    line=dict(color=colors[i % len(colors)], width=3),
                                    marker=dict(size=8)
                                ))
                            
                            fig.update_layout(
                                title=f"AI Model Comparison - {comparison_metric.replace('_', ' ').title()}",
                                xaxis_title="Training Round",
                                yaxis_title=comparison_metric.replace('_', ' ').title(),
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)",
                                font=dict(family="Inter", size=12),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Model performance summary
                            st.markdown("#### 📊 Performance Summary")
                            latest_iteration = client_comparison_data["iteration"].max()
                            summary_data = client_comparison_data[
                                (client_comparison_data["iteration"] == latest_iteration) &
                                (client_comparison_data["model_name"].isin(selected_models))
                            ]
                            
                            cols = st.columns(len(selected_models))
                            for i, model in enumerate(selected_models):
                                model_latest = summary_data[summary_data["model_name"] == model]
                                if not model_latest.empty:
                                    with cols[i]:
                                        value = model_latest[comparison_metric].iloc[0]
                                        st.markdown(f"""
                                        <div class="metric-card">
                                            <div class="metric-value">{value:.3f}</div>
                                            <div class="metric-label">{model}</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                        else:
                            st.info("Please select at least 2 models to compare.")
                    else:
                        st.info(f"Healthcare Partner #{selected_client} has only trained one model type. Multiple models needed for comparison.")
                else:
                    st.info(f"No model data available for Healthcare Partner #{selected_client}.")

        st.markdown('</div>', unsafe_allow_html=True)

    with tab5:
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        st.markdown("### 🌍 Federated Learning System Overview")
        st.markdown("""
        This dashboard shows the overall health and performance of our federated learning network, 
        including all participating healthcare partners and their latest AI model performance.
        """)

        if client_metrics_df.empty:
            st.markdown("""
            <div class="info-card">
                <h4>🌐 System Initializing</h4>
                <p>The federated learning system is starting up. System overview will be available once healthcare partners begin training.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            latest_iteration = client_metrics_df["iteration"].max()
            latest_system_data = client_metrics_df[
                client_metrics_df["iteration"] == latest_iteration
            ].copy()

            if not latest_system_data.empty:
                st.markdown(f"#### 📅 System Status (Training Round {latest_iteration})")
                
                # System metrics overview
                overview_metrics = [col for col in latest_system_data.columns if col not in ["client_id", "iteration", "model_name", "fit_status"]]
                
                selected_overview_metric = st.selectbox(
                    "📈 System Performance Metric:",
                    options=overview_metrics,
                    index=overview_metrics.index("accuracy") if "accuracy" in overview_metrics else 0,
                    help="Select which metric to display across all healthcare partners"
                )

                if selected_overview_metric:
                    # Create system overview chart
                    chart_data = latest_system_data.sort_values(by=['client_id', 'model_name'])
                    
                    fig = px.bar(
                        chart_data,
                        x=[f"Partner {row['client_id']}\n({row['model_name']})" for _, row in chart_data.iterrows()],
                        y=selected_overview_metric,
                        title=f"System-wide {selected_overview_metric.replace('_', ' ').title()} Performance",
                        color=selected_overview_metric,
                        color_continuous_scale="Viridis"
                    )
                    
                    fig.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter", size=12),
                        xaxis_title="Healthcare Partners",
                        yaxis_title=selected_overview_metric.replace('_', ' ').title(),
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # System statistics
                    st.markdown("#### 📊 Network Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_partners = latest_system_data['client_id'].nunique()
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{total_partners}</div>
                            <div class="metric-label">Healthcare Partners</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        total_models = len(latest_system_data)
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{total_models}</div>
                            <div class="metric-label">Active AI Models</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        avg_performance = latest_system_data[selected_overview_metric].mean()
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{avg_performance:.3f}</div>
                            <div class="metric-label">Average {selected_overview_metric.title()}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        best_performance = latest_system_data[selected_overview_metric].max()
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{best_performance:.3f}</div>
                            <div class="metric-label">Best {selected_overview_metric.title()}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Training status overview
                    st.markdown("#### 🔄 Training Status Overview")
                    status_summary = latest_system_data.groupby('fit_status').size().to_dict()
                    
                    status_cols = st.columns(len(status_summary))
                    for i, (status, count) in enumerate(status_summary.items()):
                        with status_cols[i]:
                            status_class = "status-success" if status == "completed" else "status-warning"
                            st.markdown(f"""
                            <div style="text-align: center; padding: 1rem;">
                                <span class="status-badge {status_class}" style="font-size: 1.1rem;">
                                    {count} models {status}
                                </span>
                            </div>
                            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 2rem 0;">
        <p>🩺 <strong>MedAI Federated Learning Platform</strong></p>
        <p style="font-size: 0.9rem;">Advancing healthcare AI while preserving patient privacy | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()