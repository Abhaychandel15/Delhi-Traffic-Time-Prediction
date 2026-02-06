from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# ======================
# LOAD MODEL & PIPELINE
# ======================
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

# ======================
# LOAD DATA (ONLY FOR SUGGESTIONS)
# ======================
df = pd.read_csv("real_features.csv")

# ======================
# COLUMN NAMES (EXACT CSV NAMES)
# ======================
COL = {
    "start": "start_area",
    "end": "end_area",
    "distance": "distance_km",
    "time": "time_of_day",
    "day": "day_of_week",
    "weather": "weather_condition",
    "traffic": "traffic_density_level",
    "road": "road_type",
    "speed": "average_speed_kmph"
}

# ======================
# OPTIONAL AREA SUGGESTIONS (NOT LIMIT)
# ======================
known_areas = sorted(
    pd.concat([df[COL["start"]], df[COL["end"]]])
    .dropna()
    .unique()
)

# ======================
# DROPDOWNS (SAFE CATEGORICALS)
# ======================
time_of_day = sorted(df[COL["time"]].dropna().unique())
days = sorted(df[COL["day"]].dropna().unique())
weather = sorted(df[COL["weather"]].dropna().unique())
traffic_density = sorted(df[COL["traffic"]].dropna().unique())
road_types = sorted(df[COL["road"]].dropna().unique())

# ======================
# HOME
# ======================
@app.route("/")
def home():
    return render_template(
        "index.html",
        known_areas=known_areas,
        time_of_day=time_of_day,
        days=days,
        weather=weather,
        traffic_density=traffic_density,
        road_types=road_types
    )

# ======================
# PREDICT
# ======================
@app.route("/predict", methods=["POST"])
def predict():
    data = {
        COL["start"]: request.form["start_area"],      # ANY AREA
        COL["end"]: request.form["end_area"],          # ANY AREA
        COL["distance"]: float(request.form["distance"]),
        COL["time"]: request.form["time_of_day"],
        COL["day"]: request.form["day"],
        COL["weather"]: request.form["weather"],
        COL["traffic"]: request.form["traffic_density"],
        COL["road"]: request.form["road_type"],
        COL["speed"]: float(request.form["avg_speed"]),
    }

    input_df = pd.DataFrame([data])

    transformed = pipeline.transform(input_df)
    prediction = model.predict(transformed)[0]

    return render_template(
        "result.html",
        prediction=round(prediction, 2)
    )

# ======================
if __name__ == "__main__":
    app.run(debug=True)
