from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import pandas as pd
from collections import Counter

def train_model(X, y, use_smote=True):
    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    if use_smote:
        class_counts = Counter(y_train)
        min_samples = min(class_counts.values())
        k_neighbors = max(min(min_samples - 1, 3), 1)

        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train.columns = X_train.columns.astype(str)

        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        # Recreate DataFrame with correct column names
        X_train = pd.DataFrame(X_train_res, columns=X_train.columns)
        y_train = pd.Series(y_train_res)

    # Ensure columns match
    X_test = X_test[X_train.columns]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss',
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f}")

    return model, X_test, y_test
