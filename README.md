# 🚦 Delhi Traffic Travel Time Prediction

An end-to-end Machine Learning project that predicts **travel time (in minutes)** for Delhi city trips based on distance, speed, traffic conditions, weather, and road type.

This project is built with a **production-ready ML pipeline**, handling preprocessing, model training, persistence, and inference on unseen data.

---

## 📌 Problem Statement

Urban traffic congestion makes travel time prediction challenging.  
This project aims to predict **trip travel time** using historical traffic data and contextual features such as:
- Distance
- Average speed
- Traffic density
- Time of day
- Weather conditions
- Road type

---

## 🧠 Solution Overview

- Stratified sampling based on distance bins to avoid bias
- Separate preprocessing for numerical and categorical features
- Robust ML pipeline using `ColumnTransformer`
- Random Forest Regressor for non-linear relationships
- Model & pipeline persistence using `joblib`
- Inference support for unseen data without retraining

---

## 📊 Dataset Description

### Input Features
- `start_area`
- `end_area`
- `distance_km`
- `time_of_day`
- `day_of_week`
- `weather_condition`
- `traffic_density_level`
- `road_type`
- `average_speed_kmph`

### Target
- `travel_time_minutes`

> ⚠️ Note:  
> Only **sample / representative data** is included in the repository to demonstrate structure and schema.  
> Full datasets follow the same format.

---

## ⚙️ Machine Learning Pipeline

### Numerical Features
- Missing value handling → `SimpleImputer (median)`
- Scaling → `StandardScaler`

### Categorical Features
- Missing value handling → `SimpleImputer (most_frequent)`
- Encoding → `OneHotEncoder (handle_unknown="ignore")`

### Model
- `RandomForestRegressor`

All preprocessing and the model are combined into a **single pipeline** to prevent data leakage and ensure consistent inference.

---

## 🗂️ Project Structure

Delhi-Traffic-Time-Prediction/
│
├── main.py
├── check_model_error.py
├── requirements.txt
├── README.md
├── data/
│ └── sample_data.csv
└── .gitignore

---

## ▶️ How to Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt

python main.py
If the model does not exist → it will train automatically

If the model exists → it will run inference on unseen data

📈 Model Evaluation

Models experimented:

Linear Regression

Decision Tree Regressor

Random Forest Regressor ✅ (Best performance)

Evaluation Metric:

Root Mean Squared Error (RMSE)

Random Forest showed the best generalization performance.

🚀 Key Highlights

End-to-end ML workflow

Clean separation of training and inference

Handles missing values and unseen categories

Production-style pipeline design

Ready for real-world deployment

🛠️ Tech Stack

Python

Pandas, NumPy

Scikit-learn

Random Forest

Joblib

Git & GitHub

👤 Author

Abhay Chandel
Aspiring Data Scientist / Machine Learning Engineer
