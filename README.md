# 🩺 NeuroNode - Federated Diabetes Predictor

## 🧠 Overview
**NeuroNode** is an innovative web application designed to predict diabetes risk using a **Federated Learning** approach. This system allows multiple healthcare providers to collaboratively train a powerful AI model **without sharing sensitive patient data** directly. By keeping data localized and only exchanging model updates, NeuroNode ensures **patient privacy** while leveraging diverse datasets to achieve **more accurate and robust predictions**.

This repository contains the **Streamlit frontend application** for NeuroNode, providing a **user-friendly interface** for health assessment, monitoring federated model performance, and gaining insights into the contributions of various healthcare partners.

---

## ✨ Features

- 🔐 **Secure Login**: Authenticated access to the dashboard.
- 🧾 **Personal Health Assessment**: Users can input their health parameters to receive a personalized diabetes risk assessment powered by the federated AI model.
- 📈 **Global Model Performance Tracking**: Visualize the accuracy, loss, and other metrics of the global federated model as it improves over training rounds.
- 🏥 **Healthcare Partner Insights**: Monitor the individual contributions and performance of participating healthcare partners.
- 🔒 **Privacy-Preserving**: Built on the principles of federated learning, ensuring no raw patient data ever leaves its original source.
- 📊 **Interactive Visualizations**: Utilizes Plotly for dynamic and informative charts.

---

## 🚀 Technologies Used

### Frontend

- [Streamlit](https://streamlit.io/) (Python web framework)
- HTML/CSS (for custom styling and enhanced UI)
- Plotly (for interactive data visualizations)
- Pandas (for data handling)
- Requests (for API communication)

### Backend

- FastAPI (Python backend)

> 📢 **Note**: This application expects an API endpoint at `http://localhost:8000/`.

Backend code: [NeuroNode Backend Repository](https://github.com/Bishnu-alt/Neronode-hackathon-)

---

## ⚙️ Installation and Setup

### ✅ Prerequisites

- Python 3.8+
- `pip` (Python package installer)

### 🛠 Steps

#### 1. Clone the Repository

```bash
git clone https://github.com/Bishnu-alt/Neronode-hackathon-).git
```
2. Set Up Python Environment (Recommended)
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install Dependencies
```
pip install -r requirements.txt
```


4. Run the Backend Server
Ensure the backend server is running at:
```
http://localhost:8000/
```

Expected Endpoints:

```
POST http://localhost:8000/         # For diabetes prediction
GET  http://localhost:8000/metrics/ # For global and client metrics
```
5. Run the Streamlit Frontend
```
streamlit run dashboard/app.py
```
The application will open in your browser at http://localhost:8501

🖥️ Usage
🔐 Login
Default demo credentials:

```
Username: admin
Password: admin123
```

📊 Navigate Tabs
Health Check: Input your health data for a diabetes risk assessment.

AI Model Progress: View global model performance over federated iterations.

Healthcare Partners: Check individual performance metrics of each client.

Model Comparison




## 📄 License

This project is licensed under **Team NeuroNode**.  
All rights reserved by the contributors of Team NeuroNode.
