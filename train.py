import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load your data
data = pd.read_csv(r"C:\Users\antos\OneDrive\Desktop\AI IN ENTERPRISE SYSTEMS\LAB-5\Fish.csv")

# Separate features and target
X = data.drop('Species', axis=1)
y = data['Species']

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train the model
model = RandomForestClassifier()
model.fit(X, y_encoded)

# Save the model and label encoder
joblib.dump(model, 'fish_species_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')