
import sys
import os

# Get the directory of the current script (app.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (which is the project root 'federated learning')
project_root = os.path.join(current_dir, '..')
# Add the project root to sys.path
sys.path.append(project_root)

# Now your imports should work
from backend.api.schemas import Features, Prediction
from backend.db.connection import get_db_connection
from backend.model.fetch import fetch_global_model
# ... rest of your app.py code
from fastapi import APIRouter, HTTPException
from backend.api.schemas import Features, Prediction
from backend.db.connection import get_db_connection
from backend.model.fetch import fetch_global_model
import pandas as pd

router = APIRouter()

try:
    model = fetch_global_model(model_id=1)
except Exception as e:
    raise RuntimeError(f"Error loading global model from DB: {e}")

COLUMNS = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]

@router.post("/", response_model=Prediction)
def predict_diabetes(data: Features):
    if len(data.features) != len(COLUMNS):
        raise HTTPException(status_code=400, detail=f"Expected {len(COLUMNS)} features")

    df = pd.DataFrame([data.features], columns=COLUMNS)
    try:
        prediction = int(model.predict(df)[0])
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@router.get("/metrics/")
def get_metrics():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT client_id, round_num AS iteration, 
                   accuracy, macro_f1, recall_minority, 
                   f1_minority, f1_majority
            FROM client_updates 
            ORDER BY iteration
        """)
        records = cursor.fetchall()
        df = pd.DataFrame(records)

        if df.empty:
            return {"global": [], "clients": []}

        global_metrics = df.groupby("iteration").mean(numeric_only=True).reset_index()
        client_metrics = df.copy()

        return {
            "global": global_metrics.to_dict(orient="records"),
            "clients": client_metrics.to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching metrics: {e}")

@router.get("/accuracy/clients")
def get_client_accuracy():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT client_id, round_num AS iteration, accuracy 
            FROM client_updates 
            ORDER BY round_num
        """)
        rows = cursor.fetchall()
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching client accuracy: {e}")
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

st.set_page_config(page_title="Federated Diabetes Predictor", layout="wide")

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
    cursor: pointer;
}
.stButton>button:hover {
    background-color: #0066cc;
}
.stPlotlyChart {
    width: 100% !important;
}
.stPlotlyChart > div {
    width: 100% !important;
}
.stMarkdown h3 {
    color: #36454F;
    margin-bottom: 1em;
}
</style>
""", unsafe_allow_html=True)

def login():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔐 Login to Access the Dashboard")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.success("✅ Login successful!")
            st.rerun()
        else:
            st.error("❌ Incorrect credentials")
    st.markdown('</div>', unsafe_allow_html=True)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

with st.sidebar:
    st.title("⚙️ Options")
    st.info("👤 Logged in as: admin")
    if st.button("🔓 Logout"):
        st.session_state.logged_in = False
        st.rerun()

st.markdown('<div class="main-title">🩺 Federated Diabetes Prediction Dashboard</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Predict", "📊 Global Model Progress", "👥 Client-wise Model Progress (Individual)", "📈 Client Model Comparison", "🌍 Client Performance Overview"])

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    with st.expander("🧪 Fill Your Health Details to Predict Diabetes Risk", expanded=True):
        yes_no = {"No": 0.0, "Yes": 1.0}
        gender_map = {"Male": 1.0, "Female": 0.0}
        gen_health_map = {"Excellent": 1.0, "Very Good": 2.0, "Good": 3.0, "Fair": 4.0, "Poor": 5.0}
        
        age_map_brfss = {
            "18-24": 1.0, "25-29": 2.0, "30-34": 3.0, "35-39": 4.0,
            "40-44": 5.0, "45-49": 6.0, "50-54": 7.0, "55-59": 8.0,
            "60-64": 9.0, "65-69": 10.0, "70-74": 11.0, "75-79": 12.0,
            "80+": 13.0
        }

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
                highbp = st.radio("High Blood Pressure", list(yes_no.keys()), index=0)
                highchol = st.radio("High Cholesterol", list(yes_no.keys()), index=0)
                cholcheck = st.radio("Had Cholesterol Check in 5 Years", list(yes_no.keys()), index=0)
                bmi = st.slider("Body Mass Index (BMI)", 10.0, 50.0, 25.0, help="Your Body Mass Index. A value between 18.5 and 24.9 is considered healthy.")
                smoker = st.radio("Smoked at Least 100 Cigarettes in Lifetime", list(yes_no.keys()), index=0)
                stroke = st.radio("Ever Had a Stroke", list(yes_no.keys()), index=0)
                heart = st.radio("Ever Had Heart Disease or Attack", list(yes_no.keys()), index=0)
                phys = st.radio("Physical Activity in Past 30 Days (not job)", list(yes_no.keys()), index=0)
                fruits = st.radio("Consume Fruit 1 or More Times Per Day", list(yes_no.keys()), index=0)
                veggies = st.radio("Consume Vegetables 1 or More Times Per Day", list(yes_no.keys()), index=0)
                alcohol = st.radio("Heavy Alcohol Consumption (men >=14, women >=7 drinks/week)", list(yes_no.keys()), index=0)

            with col2:
                healthcare = st.radio("Has Any Kind of Health Care Coverage", list(yes_no.keys()), index=0)
                nodoc = st.radio("Could Not Afford to See Doctor Due to Cost", list(yes_no.keys()), index=0)
                genhlth = st.selectbox("General Health", list(gen_health_map.keys()), index=2)
                ment = st.slider("Days of Poor Mental Health in Past 30 Days", 0.0, 30.0, 0.0, help="Number of days during the past 30 days that your mental health was not good.")
                physhlth = st.slider("Days of Poor Physical Health in Past 30 Days", 0.0, 30.0, 0.0, help="Number of days during the past 30 days that your physical health was not good.")
                walk = st.radio("Difficulty Walking or Climbing Stairs", list(yes_no.keys()), index=0)
                sex = st.radio("Sex", list(gender_map.keys()), index=0)
                age = st.selectbox("Age Range", list(age_map_brfss.keys()), index=len(age_map_brfss)//2)
                edu = st.selectbox("Education Level", list(education_map.keys()), index=3)
                income = st.selectbox("Income Level", list(income_map.keys()), index=len(income_map)//2)

            submitted = st.form_submit_button("🚀 Predict My Diabetes Risk")

            if submitted:
                features = [
                    yes_no[highbp], yes_no[highchol], yes_no[cholcheck], bmi,
                    yes_no[smoker], yes_no[stroke], yes_no[heart], yes_no[phys],
                    yes_no[fruits], yes_no[veggies], yes_no[alcohol], yes_no[healthcare],
                    yes_no[nodoc], gen_health_map[genhlth], ment, physhlth,
                    yes_no[walk], gender_map[sex], age_map_brfss[age], education_map[edu],
                    income_map[income],
                ]
                try:
                    with st.spinner("Predicting your diabetes risk..."):
                        response = requests.post("http://localhost:8000/", json={"features": features})
                        if response.status_code == 200:
                            prediction = response.json()["prediction"]
                            if prediction == 1:
                                st.error("⚠️ Based on the provided information, you are at a **HIGH RISK** of diabetes. Please consult a doctor for a professional diagnosis and advice.")
                            else:
                                st.success("✅ Based on the provided information, you are at a **LOW RISK** of diabetes. Keep up the healthy habits!")
                        else:
                            st.warning(f"Error from server ({response.status_code}): {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("🚨 Could not connect to the backend server. Please ensure the backend is running at http://localhost:8000.")
                except Exception as e:
                    st.error(f"An unexpected error occurred during prediction: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

@st.cache_data(ttl=600)
def fetch_metrics():
    try:
        res = requests.get("http://localhost:8000/metrics/")
        res.raise_for_status()
        data = res.json()
        
        global_df = pd.DataFrame(data.get("global", [])) # <-- Change this
        client_df = pd.DataFrame(data.get("clients", [])) 

        if not global_df.empty:
            global_df["iteration"] = global_df["iteration"].astype(int)
            global_df = global_df.sort_values("iteration").set_index("iteration")
        if not client_df.empty:
            client_df["iteration"] = client_df["iteration"].astype(int)
            client_df["client_id"] = client_df["client_id"].astype(int)
            client_df["model_name"] = client_df["model_name"].astype(str)
            client_df = client_df.sort_values(["client_id", "model_name", "iteration"])
        return global_df, client_df
    except requests.exceptions.ConnectionError:
        st.error("🚨 Could not connect to the backend server. Please ensure the backend is running at http://localhost:8000.")
        return pd.DataFrame(), pd.DataFrame()
    except requests.exceptions.HTTPError as e:
        st.error(f"Error fetching data from backend: {e.response.status_code} - {e.response.text}")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching metrics: {e}")
        return pd.DataFrame(), pd.DataFrame()

global_metrics_df, client_metrics_df = fetch_metrics()

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Global Model Metrics Over Federated Iterations")

    if global_metrics_df.empty:
        st.warning("No global metrics available yet. Run some federated learning rounds on your backend.")
    else:
        metric_options_global = [col for col in global_metrics_df.columns if col not in ['iteration']]
        
        default_global_metric_index = metric_options_global.index("accuracy") if "accuracy" in metric_options_global else 0
        
        selected_global_metric = st.selectbox(
            "Select a metric to visualize for Global Model:",
            metric_options_global,
            index=default_global_metric_index,
            key="global_metric_select"
        )
        
        chart_type_global = st.radio(
            "Choose Chart Type for Global Model:",
            ("Line Chart", "Bar Chart"),
            key="global_chart_type"
        )

        if selected_global_metric:
            fig, ax = plt.subplots(figsize=(10, 5)) 
            
            if chart_type_global == "Line Chart":
                ax.plot(global_metrics_df.index, global_metrics_df[selected_global_metric], marker='o', linestyle='-', color='#1E90FF')
            elif chart_type_global == "Bar Chart":
                ax.bar(global_metrics_df.index, global_metrics_df[selected_global_metric], color='#1E90FF', width=0.8)
            
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.set_xlabel("Federated Learning Iteration", fontsize=12)
            ax.set_ylabel(selected_global_metric.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f"Global Model {selected_global_metric.replace('_', ' ').title()} Over Iterations", fontsize=14, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Please select a metric to display the global model's progress.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("👥 Client-wise Model Progress (Individual View)")

    if client_metrics_df.empty:
        st.warning("No client metrics available yet. Run some federated learning rounds on your backend.")
    else:
        unique_clients = sorted(client_metrics_df["client_id"].unique())

        for client_id in unique_clients:
            with st.expander(f"📍 View Progress for Client {client_id}", expanded=False):
                client_df_all_models = client_metrics_df[client_metrics_df["client_id"] == client_id].copy()
                
                if client_df_all_models.empty:
                    st.info(f"No metrics available for Client {client_id}.")
                    continue
                
                numeric_cols_for_plot = [col for col in client_df_all_models.columns if col not in ["client_id", "iteration", "model_name", "fit_status"]]
                
                default_metric_index_individual = numeric_cols_for_plot.index("accuracy") if "accuracy" in numeric_cols_for_plot else 0

                col_select_metric, col_select_model, col_chart_type = st.columns([0.35, 0.35, 0.3])

                with col_select_metric:
                    selected_metric_individual = st.selectbox(
                        f"Select a metric for Client {client_id}:",
                        numeric_cols_for_plot,
                        index=default_metric_index_individual,
                        key=f"client_individual_{client_id}_metric_select"
                    )
                
                with col_select_model:
                    unique_models_for_client = sorted(client_df_all_models["model_name"].unique())
                    selected_model_individual = st.selectbox(
                        f"Select Model Name for Client {client_id}:",
                        unique_models_for_client,
                        key=f"client_individual_{client_id}_model_select"
                    )

                with col_chart_type:
                    chart_type_client_individual = st.radio(
                        f"Choose Chart Type for Client {client_id}:",
                        ("Line Chart", "Bar Chart"),
                        key=f"client_individual_{client_id}_chart_type"
                    )

                if selected_metric_individual and selected_model_individual:
                    client_df_filtered = client_df_all_models[client_df_all_models["model_name"] == selected_model_individual].copy()
                    client_df_filtered.set_index("iteration", inplace=True)
                    
                    if client_df_filtered.empty:
                        st.info(f"No data for selected model '{selected_model_individual}' for Client {client_id}.")
                        continue

                    fig, ax = plt.subplots(figsize=(10, 5))
                    
                    if chart_type_client_individual == "Line Chart":
                        ax.plot(client_df_filtered.index, client_df_filtered[selected_metric_individual], 
                                marker='o', linestyle='--', color='#FF4B4B', 
                                label=f"{selected_model_individual} (Client {client_id})")
                    elif chart_type_client_individual == "Bar Chart":
                        ax.bar(client_df_filtered.index, client_df_filtered[selected_metric_individual], 
                                color='#FF4B4B', width=0.8, 
                                label=f"{selected_model_individual} (Client {client_id})")
                    
                    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                    ax.set_xlabel("Federated Learning Iteration", fontsize=12)
                    ax.set_ylabel(selected_metric_individual.replace('_', ' ').title(), fontsize=12)
                    ax.set_title(f"Client {client_id} - Model '{selected_model_individual}' {selected_metric_individual.replace('_', ' ').title()} Over Iterations", fontsize=14, fontweight='bold')
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    st.subheader(f"🔍 Fit Status for Model '{selected_model_individual}' (Client {client_id})")
                    fit_status_data = client_df_filtered[['fit_status']].reset_index().drop_duplicates(subset=['iteration'])
                    if not fit_status_data.empty:
                        st.dataframe(fit_status_data.set_index('iteration'))
                    else:
                        st.info(f"No fit status data for model '{selected_model_individual}' for Client {client_id}.")
                else:
                    st.info(f"Please select a metric and model to display Client {client_id}'s progress.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Compare Different Local Models within a Selected Client")

    if client_metrics_df.empty:
        st.warning("No client metrics available for comparison. Run some federated learning rounds on your backend.")
    else:
        unique_clients_for_comparison_tab = sorted(client_metrics_df["client_id"].unique())
        
        selected_client_for_model_comparison = st.selectbox(
            "Select a client to compare their models:",
            options=unique_clients_for_comparison_tab,
            key="compare_single_client_select"
        )

        if selected_client_for_model_comparison:
            client_data_for_comparison = client_metrics_df[
                client_metrics_df["client_id"] == selected_client_for_model_comparison
            ].copy()

            if client_data_for_comparison.empty:
                st.info(f"No model data available for Client {selected_client_for_model_comparison}.")
            else:
                available_model_names = sorted(client_data_for_comparison["model_name"].unique())
                
                default_models = available_model_names[:min(2, len(available_model_names))]
                if len(available_model_names) > 0 and not default_models:
                     default_models = [available_model_names[0]]

                selected_model_names_to_compare = st.multiselect(
                    f"Select models (by name) for Client {selected_client_for_model_comparison} to compare:",
                    options=available_model_names,
                    default=default_models,
                    key="compare_models_multiselect"
                )

                if not selected_model_names_to_compare:
                    st.info("Please select at least one model name to compare for this client.")
                else:
                    metrics_for_plotting = [
                        col for col in client_data_for_comparison.columns 
                        if col not in ["client_id", "iteration", "model_name", "fit_status"]
                    ]
                    
                    default_metric_index_model_comp = metrics_for_plotting.index("accuracy") if "accuracy" in metrics_for_plotting else 0

                    selected_metric_for_model_comparison = st.selectbox(
                        "Select a metric to compare across these models:",
                        options=metrics_for_plotting,
                        index=default_metric_index_model_comp,
                        key="compare_model_metric_select"
                    )

                    chart_type_model_comparison = st.radio(
                        "Choose Chart Type for Model Comparison:",
                        ("Line Chart", "Bar Chart"),
                        key="model_comparison_chart_type"
                    )

                    if selected_metric_for_model_comparison:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        colors = plt.cm.get_cmap('tab10', len(selected_model_names_to_compare))
                        
                        if chart_type_model_comparison == "Line Chart":
                            for i, model_name in enumerate(selected_model_names_to_compare):
                                model_df = client_data_for_comparison[
                                    client_data_for_comparison["model_name"] == model_name
                                ].sort_values("iteration")
                                if not model_df.empty:
                                    ax.plot(model_df["iteration"], model_df[selected_metric_for_model_comparison],
                                             marker='o', linestyle='-', label=f"Model: {model_name}", color=colors(i))
                            
                            ax.set_xlabel("Federated Learning Iteration", fontsize=12)
                            ax.set_title(f"Client {selected_client_for_model_comparison}: {selected_metric_for_model_comparison.replace('_', ' ').title()} Trend for Different Models", fontsize=14, fontweight='bold')
                            ax.legend(title="Model Name")
                            
                        elif chart_type_model_comparison == "Bar Chart":
                            latest_iteration = client_data_for_comparison["iteration"].max()
                            bar_data = []
                            labels = []
                            bar_colors = []
                            
                            latest_data_subset = client_data_for_comparison[
                                (client_data_for_comparison["iteration"] == latest_iteration) &
                                (client_data_for_comparison["model_name"].isin(selected_model_names_to_compare))
                            ].sort_values("model_name")

                            if not latest_data_subset.empty:
                                for i, model_name in enumerate(selected_model_names_to_compare):
                                    model_latest_metric = latest_data_subset[latest_data_subset["model_name"] == model_name][selected_metric_for_model_comparison]
                                    if not model_latest_metric.empty:
                                        bar_data.append(model_latest_metric.iloc[0])
                                        labels.append(model_name)
                                        bar_colors.append(colors(i))
                                
                                if bar_data:
                                    x = np.arange(len(labels))
                                    ax.bar(x, bar_data, color=bar_colors, width=0.7)
                                    ax.set_xticks(x)
                                    ax.set_xticklabels(labels, rotation=45, ha='right')
                                    ax.set_xlabel("Model Name", fontsize=12)
                                    ax.set_title(f"Comparison of {selected_metric_for_model_comparison.replace('_', ' ').title()} at Iteration {latest_iteration} for Client {selected_client_for_model_comparison}", fontsize=14, fontweight='bold')
                                else:
                                    st.info("No data to plot for the selected models at the latest iteration.")
                                    plt.close(fig)
                                    fig = None
                            else:
                                st.info(f"No data for selected models at the latest iteration ({latest_iteration}).")
                                plt.close(fig)
                                fig = None

                        if fig:
                            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                            ax.set_ylabel(selected_metric_for_model_comparison.replace('_', ' ').title(), fontsize=12)
                            ax.grid(True, linestyle='--', alpha=0.7)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                        
                        st.subheader(f"🔍 Fit Status for Models (Client {selected_client_for_model_comparison})")
                        fit_status_comp_df = client_data_for_comparison[
                            client_data_for_comparison["model_name"].isin(selected_model_names_to_compare)
                        ].pivot_table(index='iteration', columns='model_name', values='fit_status', aggfunc='first')

                        if not fit_status_comp_df.empty:
                            st.dataframe(fit_status_comp_df)
                        else:
                            st.info("No fit status data for the selected models within this client.")
                    else:
                        st.info("Please select a metric to compare across models.")
        else:
            st.info("Please select a client to enable model comparison.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab5:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🌍 Client Performance Overview (Latest Iteration)")

    if client_metrics_df.empty:
        st.warning("No client metrics available for overview. Run some federated learning rounds on your backend.")
    else:
        latest_iteration_overall = client_metrics_df["iteration"].max()
        
        latest_client_metrics = client_metrics_df[
            client_metrics_df["iteration"] == latest_iteration_overall
        ].copy()

        if latest_client_metrics.empty:
            st.info(f"No client data found for the latest iteration: {latest_iteration_overall}. This might happen if clients haven't reported yet.")
        else:
            st.write(f"Displaying performance for the **latest iteration: {latest_iteration_overall}**")

            overview_metric_options = [col for col in latest_client_metrics.columns if col not in ["client_id", "iteration", "model_name", "fit_status"]]
            default_overview_metric_index = overview_metric_options.index("accuracy") if "accuracy" in overview_metric_options else 0

            selected_overview_metric = st.selectbox(
                "Select a metric to overview client performance:",
                options=overview_metric_options,
                index=default_overview_metric_index,
                key="overview_metric_select"
            )

            if selected_overview_metric:
                fig, ax = plt.subplots(figsize=(12, 6))

                plot_data = latest_client_metrics.sort_values(by=['client_id', 'model_name'])

                if not plot_data.empty:
                    labels = [f"Client {row['client_id']} ({row['model_name']})" for idx, row in plot_data.iterrows()]
                    values = plot_data[selected_overview_metric].values
                    client_ids_for_ticks = [f"Client {c_id}" for c_id in plot_data['client_id'].unique()]

                    x = np.arange(len(labels))
                    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))

                    ax.bar(x, values, color=colors)
                    ax.set_xticks(x)
                    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
                    
                    ax.set_xlabel("Client ID (Model Name)", fontsize=12)
                    ax.set_ylabel(selected_overview_metric.replace('_', ' ').title(), fontsize=12)
                    ax.set_title(f"Client Performance for {selected_overview_metric.replace('_', ' ').title()} (Latest Iteration {latest_iteration_overall})", fontsize=14, fontweight='bold')
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("No data to display for the selected metric at the latest iteration.")
            else:
                st.info("Please select a metric to view the client performance overview.")
    st.markdown('</div>', unsafe_allow_html=True)