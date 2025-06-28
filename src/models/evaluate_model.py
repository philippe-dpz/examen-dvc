import pandas as pd
import joblib
import json
import os
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # Charger les données de test
    X_test = pd.read_csv("data/processed/X_test_scaled.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    # Charger le modèle entraîné
    model = joblib.load("models/trained_model.joblib")

    # Faire les prédictions
    y_pred = model.predict(X_test)

    # Calculer les métriques
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    # Afficher les résultats
    print(f"✅ MSE: {mse:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

    # Créer le dossier metrics s'il n'existe pas
    os.makedirs("metrics", exist_ok=True)

    # Sauvegarder les scores
    with open("metrics/accuracy.json", "w") as f:
        json.dump({"mse": mse, "rmse": rmse, "r2": r2}, f, indent=4)

    # Sauvegarder les prédictions dans un fichier CSV
    os.makedirs("data", exist_ok=True)
    predictions_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    predictions_df.to_csv("data/predictions.csv", index=False)

if __name__ == "__main__":
    main()
