"""Script to train the RandomForest crop recommendation model."""
import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('Data-processed/crop_recommendation.csv')
features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']

Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, target, test_size=0.2, random_state=2)

# Train RandomForest
RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain, Ytrain)

# Create models directory in app folder
os.makedirs('app/models', exist_ok=True)

# Save model
with open('app/models/RandomForest.pkl', 'wb') as f:
    pickle.dump(RF, f)

print("RandomForest model trained and saved to app/models/RandomForest.pkl")
