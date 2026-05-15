from pathlib import Path
import pickle

import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "house_price_model.pkl"

FEATURES = [
    {"name": "MedInc", "label": "Median income", "hint": "Income in tens of thousands of dollars", "step": "0.01"},
    {"name": "HouseAge", "label": "House age", "hint": "Average house age in years", "step": "1"},
    {"name": "AveRooms", "label": "Average rooms", "hint": "Average number of rooms per household", "step": "0.01"},
    {"name": "AveBedrms", "label": "Average bedrooms", "hint": "Average number of bedrooms per household", "step": "0.01"},
    {"name": "Population", "label": "Population", "hint": "Block population", "step": "1"},
    {"name": "AveOccup", "label": "Average occupancy", "hint": "Average household size", "step": "0.01"},
    {"name": "Latitude", "label": "Latitude", "hint": "Location latitude", "step": "0.0001"},
    {"name": "Longitude", "label": "Longitude", "hint": "Location longitude", "step": "0.0001"},
]

DEFAULT_VALUES = {
    "MedInc": 3.8,
    "HouseAge": 29,
    "AveRooms": 5.0,
    "AveBedrms": 1.0,
    "Population": 1000,
    "AveOccup": 3.0,
    "Latitude": 34.05,
    "Longitude": -118.24,
}

EDA_SUMMARY = {
    "dataset_rows": "20,640",
    "feature_count": "8 numeric features",
    "target": "Median house value in $100k units",
    "missing": "No missing values in the model inputs",
    "split": "80/20 train-test split with random_state=42",
}

MODEL_SUMMARY = {
    "type": "Linear Regression",
    "artifact": "house_price_model.pkl",
    "purpose": "Predict California housing prices from block-level census features",
}

METRICS = {
    "mae": 0.5332001304956558,
    "mse": 0.555891598695244,
    "r2": 0.5757877060324511,
    "accuracy": 57.57877060324511,
}

app = Flask(__name__)

with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)


def build_form_values(form_data=None):
    values = {}
    for feature in FEATURES:
        values[feature["name"]] = (form_data.get(feature["name"]) if form_data else None) or str(DEFAULT_VALUES[feature["name"]])
    return values


def predict_price(feature_values):
    ordered_values = [feature_values[feature["name"]] for feature in FEATURES]
    input_frame = pd.DataFrame([ordered_values], columns=[feature["name"] for feature in FEATURES])
    prediction_units = float(model.predict(input_frame)[0])
    prediction_dollars = prediction_units * 100000
    return prediction_units, prediction_dollars


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        features=FEATURES,
        form_values=build_form_values(),
        prediction=None,
        error=None,
        eda=EDA_SUMMARY,
        model_summary=MODEL_SUMMARY,
        metrics=METRICS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = {feature["name"]: float(request.form[feature["name"]]) for feature in FEATURES}
        prediction_units, prediction_dollars = predict_price(values)
        prediction = {
            "units": round(prediction_units, 3),
            "dollars": round(prediction_dollars, 2),
        }
        error = None
    except (TypeError, ValueError, KeyError):
        prediction = None
        error = "Please enter valid numeric values for every field."

    return render_template(
        "index.html",
        features=FEATURES,
        form_values=build_form_values(request.form),
        prediction=prediction,
        error=error,
        eda=EDA_SUMMARY,
        model_summary=MODEL_SUMMARY,
        metrics=METRICS,
    )


if __name__ == "__main__":
    app.run(debug=True)