"""Predict one delivery, from the command line."""

import pandas as pd

from delivery import load_model


def main():
    # Load the saved model
    model = load_model("model.joblib")

    # Build one-row DataFrame
    order = pd.DataFrame([{
        "distance_km": 7,
        "prep_time_min": 25,
        "traffic_level": 3,
        "rain": 0
    }])

    # Make prediction
    minutes = model.predict(order)[0]

    # Print prediction
    print(f"PREDICTION: {minutes:.1f}")


if __name__ == "__main__":
    raise SystemExit(main())
