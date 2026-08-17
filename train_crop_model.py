"""Simple training script for crop recommendation model.

Usage:
    python train_crop_model.py

The script expects a CSV file at `app/Data/crop_recommendation.csv` with columns:
    N,P,K,temperature,humidity,ph,rainfall,label

It trains a RandomForestClassifier and saves the model to
`app/models/crop_model.pkl` using joblib. It also prints test accuracy.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


def main():
    base_path = os.path.join('app', 'Data', 'crop_recommendation.csv')
    extended_path = os.path.join('app', 'Data', 'crop_recommendation_extended.csv')
    extra_path = os.path.join('app', 'Data', 'crop_recommendation_extra.csv')

    # prefer using the extended dataset if present
    if os.path.exists(extended_path):
        print(f"Using extended dataset at {extended_path}")
        df = pd.read_csv(extended_path)
    else:
        if not os.path.exists(base_path):
            print(f"Dataset not found at {base_path}. Please add the Crop Recommendation dataset and try again.")
            return
        # base dataset (small sample shipped with the app)
        df = pd.read_csv(base_path)

        # optional extra data file where you can append more rows
        if os.path.exists(extra_path):
            extra_df = pd.read_csv(extra_path)
            print(f"Loaded extra crop data from {extra_path} with {len(extra_df)} rows.")
            df = pd.concat([df, extra_df], ignore_index=True)

    required_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
    if not all(col in df.columns for col in required_cols):
        print(f"Dataset must contain columns: {required_cols}")
        return

    print(f"Training crop model on {len(df)} rows and {df['label'].nunique()} crops.")

    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']

    # if dataset is small or classes are imbalanced, stratify may fail
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred))

    # make sure models folder exists
    model_dir = os.path.join('app', 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'crop_model.pkl')
    joblib.dump(model, model_path)
    print(f"Trained model saved to {model_path}")


if __name__ == '__main__':
    main()
