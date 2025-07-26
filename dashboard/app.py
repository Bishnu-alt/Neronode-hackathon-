import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ------------------- Streamlit Config -------------------
st.set_page_config(page_title="Federated Diabetes Predictor", layout="wide")

# ------------------- Styling -------------------
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        color: #2F4F4F;
        text-align: center;
        margin-bottom: 0.5em;
    }
    .card {
        background-color: #ffffff;
        padding: 2em;
        border-radius: 16px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 1.5em;
    }
    .stButton>button {
        background-color: #1E90FF;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1.5em;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0066cc;
    }
</style>
""", unsafe_allow_html=True)

# ------------------- Login System -------------------
def login():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔐 Login to Access the Dashboard")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.success("✅ Login successful!")
            st.experimental_rerun()
        else:
            st.error("❌ Incorrect credentials")
    st.markdown('</div>', unsafe_allow_html=True)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# ------------------- Sidebar -------------------
with st.sidebar:
    st.title("⚙️ Options")
    st.info("👤 Logged in as: `admin`")
    if st.button("🔓 Logout"):
        st.session_state.logged_in = False
        st.rerun()

# ------------------- Main Title -------------------
st.markdown('<div class="main-title">🩺 Federated Diabetes Prediction Dashboard</div>', unsafe_allow_html=True)

# ------------------- Tabs -------------------
tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📊 Global Accuracy", "👥 Client-wise Accuracy"])

# ------------------ TAB 1: PREDICT -------------------
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    with st.expander("🧪 Fill Your Health Details to Predict Diabetes Risk", expanded=True):
        yes_no = {"No": 0.0, "Yes": 1.0}
        gender_map = {"Male": 1.0, "Female": 0.0}
        gen_health_map = {"Excellent": 1.0, "Very Good": 2.0, "Good": 3.0, "Fair": 4.0, "Poor": 5.0}
        age_map = {f"{i*5}-{i*5+4}": float(i) for i in range(1, 14)}
        education_map = {
            "Never Attended School": 1.0, "Elementary": 2.0, "Middle School": 3.0,
            "High School": 4.0, "Some College": 5.0, "Associate Degree": 6.0,
            "Bachelor’s Degree": 7.0, "Graduate Degree": 8.0
        }
        income_map = {
            "<$10K": 1.0, "$10K–$15K": 2.0, "$15K–$20K": 3.0,
            "$20K–$25K": 4.0, "$25K–$35K": 5.0, "$35K–$50K": 6.0,
            "$50K–$75K": 7.0, ">$75K": 8.0
        }

        with st.form("predict_form"):
            col1, col2 = st.columns(2)

            with col1:
                highbp = st.radio("High Blood Pressure", yes_no.keys())
                highchol = st.radio("High Cholesterol", yes_no.keys())
                cholcheck = st.radio("Cholesterol Check", yes_no.keys())
                bmi = st.slider("BMI", 10.0, 50.0, 25.0)
                smoker = st.radio("Smoker", yes_no.keys())
                stroke = st.radio("Stroke", yes_no.keys())
                heart = st.radio("Heart Disease or Attack", yes_no.keys())
                phys = st.radio("Physical Activity", yes_no.keys())
                fruits = st.radio("Consumes Fruits", yes_no.keys())
                veggies = st.radio("Consumes Vegetables", yes_no.keys())
                alcohol = st.radio("Heavy Alcohol Consumption", yes_no.keys())

            with col2:
                healthcare = st.radio("Has Healthcare Access", yes_no.keys())
                nodoc = st.radio("Could Not Afford Doctor", yes_no.keys())
                genhlth = st.selectbox("General Health", gen_health_map.keys())
                ment = st.slider("Mental Health Days", 0.0, 30.0, 5.0)
                physhlth = st.slider("Physical Health Days", 0.0, 30.0, 5.0)
                walk = st.radio("Difficulty Walking", yes_no.keys())
                sex = st.radio("Sex", gender_map.keys())
                age = st.selectbox("Age Range", age_map.keys())
                edu = st.selectbox("Education Level", education_map.keys())
                income = st.selectbox("Income Level", income_map.keys())

            submitted = st.form_submit_button("🚀 Predict Now")

            if submitted:
                features = [
                    yes_no[highbp], yes_no[highchol], yes_no[cholcheck], bmi,
                    yes_no[smoker], yes_no[stroke], yes_no[heart], yes_no[phys],
                    yes_no[fruits], yes_no[veggies], yes_no[alcohol], yes_no[healthcare],
                    yes_no[nodoc], gen_health_map[genhlth], ment, physhlth,
                    yes_no[walk], gender_map[sex], age_map[age], education_map[edu],
                    income_map[income],
                ]
                try:
                    with st.spinner("Predicting..."):
                        response = requests.post("http://localhost:8000/predict/", json={"features": features})
                    if response.status_code == 200:
                        prediction = response.json()["prediction"]
                        if prediction == 1:
                            st.error("⚠️ High risk of diabetes. Please consult a doctor.")
                        else:
                            st.success("✅ Low risk of diabetes. Stay healthy!")
                    else:
                        st.warning("Error from server: " + response.text)
                except Exception as e:
                    st.error(f"Could not connect to backend. Error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ TAB 2: GLOBAL ACCURACY -------------------
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Global Model Metrics Over Federated Iterations")
    try:
        with st.spinner("Loading global metrics..."):
            res = requests.get("http://localhost:8000/metrics/")
        if res.status_code == 200:
            global_data = res.json()["global"]
            df = pd.DataFrame(global_data)
            if df.empty:
                st.warning("No global metrics available.")
            else:
                df["iteration"] = df["iteration"].astype(int)
                df = df.sort_values("iteration")
                df.set_index("iteration", inplace=True)

                fig, ax = plt.subplots(figsize=(8, 4))
                for column in df.columns:
                    ax.plot(df.index, df[column], label=column)
                ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Metric Value")
                ax.set_title("Global Metrics vs Iteration")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.warning(f"Error fetching data: {res.text}")
    except Exception as e:
        st.error(f"Failed to fetch metrics: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ TAB 3: CLIENT-WISE ACCURACY -------------------
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("👥 Client-wise Metrics Over Iterations")

    try:
        with st.spinner("Loading client-wise metrics..."):
            res = requests.get("http://localhost:8000/metrics/")
        if res.status_code == 200:
            client_data = res.json()["clients"]
            df = pd.DataFrame(client_data)
            if df.empty:
                st.warning("No client metrics available.")
            else:
                df["iteration"] = df["iteration"].astype(int)
                for client_id in df["client_id"].unique():
                    with st.expander(f"📍 Client {client_id}", expanded=False):
                        client_df = df[df["client_id"] == client_id].sort_values("iteration")
                        client_df.set_index("iteration", inplace=True)
                        metrics_df = client_df.drop(columns=["client_id"])

                        fig, ax = plt.subplots(figsize=(8, 4))
                        for column in metrics_df.columns:
                            ax.plot(metrics_df.index, metrics_df[column], label=column)
                        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                        ax.set_xlabel("Iteration")
                        ax.set_ylabel("Metric Value")
                        ax.set_title(f"Client {client_id} Metrics")
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)
                        plt.close(fig)
        else:
            st.warning(f"Error fetching data: {res.text}")
    except Exception as e:
        st.error(f"Failed to fetch client metrics: {e}")
    st.markdown('</div>', unsafe_allow_html=True)
