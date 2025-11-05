import streamlit as st
import pandas as pd
import joblib
from src.utils import clean_text
from src.feature_engineering import generate_text_embeddings

st.set_page_config(page_title="🔧 Failure Prediction", layout="wide")
st.title("🔧 Failure Prediction with CamemBERT + XGBoost")

# Load trained model
try:
    model = joblib.load("models/xgb_model.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    st.sidebar.success("✅ Trained model, label encoder, and feature columns loaded successfully!")
except:
    st.sidebar.error("❌ Model files not found. Please train them first using main.py.")
    st.stop()
# Input form
st.subheader("📝 Enter failure details to predict type")
desc = st.text_area("Failure description:", "")
site = st.text_input("Site:")
equip = st.text_input("Equipment:")
code_equip = st.text_input("Equipment code:")
type_equip = st.text_input("Equipment type:")


if st.button("Predict"):
    if not desc.strip():
        st.warning("Please enter a failure description.")
    else:
        with st.spinner("Generating embeddings and predicting..."):
            # Clean and encode text
            input_text_df = pd.DataFrame({"Description de la Panne": [clean_text(desc)]})
            emb_df = generate_text_embeddings(input_text_df)

            # Encode structured features
            structured_input = pd.DataFrame({
                "Site": [clean_text(site)],
                "Equipement": [clean_text(equip)],
                "Code Equipement": [clean_text(code_equip)],
                "Type d'equipement": [clean_text(type_equip)]
            })
            structured_input_encoded = pd.get_dummies(structured_input)

            # Combine embeddings + structured data
            X_input = pd.concat([emb_df.reset_index(drop=True), structured_input_encoded.reset_index(drop=True)], axis=1)

            # Align with training feature columns
            X_input = X_input.reindex(columns=feature_columns, fill_value=0)

            try:
                prediction_encoded = model.predict(X_input)
                prediction = label_encoder.inverse_transform(prediction_encoded)
                st.success(f"🧠 Predicted type of failure: **{prediction[0]}**")
            except Exception as e:
                st.error(f"⚠️ Prediction failed: {e}")
