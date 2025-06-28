import pandas as pd
import joblib
import pickle
import os
from sklearn.ensemble import RandomForestRegressor

def main():
    # Charger les données
    X_train = pd.read_csv("data/processed_scaled/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

    # Charger les meilleurs paramètres
    with open("models/best_params.pkl", "rb") as f:
        best_params = pickle.load(f)

    # Créer et entraîner le modèle
    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)

    # Créer dossier models s'il n'existe pas
    os.makedirs("models", exist_ok=True)

    # Sauvegarder le modèle
    joblib.dump(model, "models/trained_model.joblib")
    print("✅ Modèle entraîné et sauvegardé dans models/trained_model.joblib")

if __name__ == "__main__":
    main()
import pandas as pd
import joblib
import pickle
import os
from sklearn.ensemble import RandomForestRegressor

def main():
    # Charger les données
    X_train = pd.read_csv("data/processed_scaled/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

    # Charger les meilleurs paramètres
    with open("models/best_params.pkl", "rb") as f:
        best_params = pickle.load(f)

    # Créer et entraîner le modèle
    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)

    # Créer dossier models s'il n'existe pas
    os.makedirs("models", exist_ok=True)

    # Sauvegarder le modèle
    joblib.dump(model, "models/trained_model.joblib")
    print("✅ Modèle entraîné et sauvegardé dans models/trained_model.joblib")

if __name__ == "__main__":
    main()
