from src.data_cleaning import load_and_clean_data
from src.feature_engineering import generate_text_embeddings, encode_structured_features, encode_target
from src.model_training import train_model
import pandas as pd
import joblib

# Load and clean data
df = load_and_clean_data("data/medis_data1.xlsx")

# Filter rare classes (at least 2 samples)
value_counts = df['Type de panne'].value_counts()
df = df[df['Type de panne'].isin(value_counts[value_counts >= 2].index)]

# Feature engineering
desc_df = generate_text_embeddings(df)
structured_df = encode_structured_features(df)

# Combine features
X = pd.concat([desc_df, structured_df], axis=1)
X.columns = X.columns.astype(str)  # ensure all column names are strings

# Encode target
y, label_encoder = encode_target(df)

# Reset index to avoid misalignment
X.reset_index(drop=True, inplace=True)
y = pd.Series(y).reset_index(drop=True)

# Train and evaluate model
model, X_test, y_test = train_model(X, y, use_smote=True)

# Ensure X_test columns are strings
X_test.columns = X_test.columns.astype(str)
# Train and evaluate model
model, X_test, y_test = train_model(X, y, use_smote=True)

# ✅ Save model, label encoder, and feature columns
joblib.dump(model, "models/xgb_model.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")
joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

print("✅ Model, label encoder, and columns saved successfully!")
