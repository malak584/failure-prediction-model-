import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder

def generate_text_embeddings(df, model_name='dangvantuan/sentence-camembert-large'):
    model = SentenceTransformer(model_name)
    print("✅ Model loaded successfully")
    embeddings = model.encode(df['Description de la Panne'].tolist(), show_progress_bar=True)
    return pd.DataFrame(embeddings, index=df.index)

def encode_structured_features(df):
    structured_cols = ['Site', 'Equipement', 'Code Equipement', "Type d'equipement"]
    structured_df = df[structured_cols]
    encoded_df = pd.get_dummies(structured_df)
    return encoded_df

def encode_target(df, target_col='Type de panne'):
    le = LabelEncoder()
    y_encoded = le.fit_transform(df[target_col])
    return y_encoded, le
