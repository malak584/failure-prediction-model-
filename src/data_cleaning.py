import pandas as pd
from .utils import clean_text

def load_and_clean_data(file_path):
    df = pd.read_excel(file_path)

 
    
    # Strip column names
    df.columns = df.columns.str.strip()
    
    # Drop rows with missing key columns
    df = df.dropna(subset=['Description de la Panne', 'Type de panne'])
    
    # Clean text columns
    text_cols = ['Description de la Panne', 'Type de panne', 'Site', 'Equipement', 'Code Equipement', "Type d'equipement"]
    for col in text_cols:
        df[col] = df[col].apply(clean_text)
    
    # Fill missing structured data
    for col in ['Site', 'Equipement', 'Code Equipement', "Type d'equipement"]:
        df[col] = df[col].fillna("Inconnu")
    
    return df
