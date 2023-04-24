# Importer les librairies
import eli5
import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from lightgbm import LGBMClassifier

# Internal imports
model = joblib.load("./models/model_final_2.joblib")
data = joblib.load("./models/df_final.pkl")
preprocess = joblib.load("./models/transformer.sav")
all_var = joblib.load("./models/all_var.joblib")

# Créer l'objet de l'app
app = FastAPI(title="Attribution de prêt")
origins = ["http://localhost:8501", "http://127.0.0.1:8501"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def test_api():
    return {"hello": "miss"}


# data endpoints
@app.get("/database")
def database():
    all_clients = data["sk_id_curr"].tolist()

    return {"values": all_clients}


# Prprocessing endpoints
@app.get("/globale_importance")
def feature_importance_globale():
    plot = eli5.show_weights(model, feature_names=all_var)

    return plot


# Prprocessing endpoints
@app.post("/importance_locale")
def feature_importance_locale(numero_client: int):
    Client_id = data[data["sk_id_curr"] == data["sk_id_curr"]]
    X_transformed = preprocess.transform(Client_id)

    X_val = pd.DataFrame(X_transformed)
    X_val.columns = all_var

    cl = data[data["sk_id_curr"] == numero_client]
    i = cl.index.values[0]

    X_val.iloc[[i]]
    plot = eli5.show_prediction(
        model, X_val.iloc[i], feature_names=all_var, show_feature_values=True
    )
    print(type(plot))
    return plot


# Prediction endpoint
@app.post("/predict")
def predict_pret(numero_client: int):
    # data.loc[numero_client,:]
    Client_id = data[data["sk_id_curr"] == numero_client]

    X_transformed = preprocess.transform(Client_id)

    proba_lgbm = model.predict_proba(X_transformed)
    df_proba_lgbm = pd.DataFrame(proba_lgbm, columns=["prob_0", "prob_1"])

    # Calcul du seuil
    diff = df_proba_lgbm["prob_1"] - df_proba_lgbm["prob_0"]
    df_proba_lgbm["seuil"] = diff
    seuil = 0.41

    if diff.values[0] <= seuil:
        diff = "Prêt accordé, client sûr"
    else:
        diff = "Prêt non accordé, risque de défaut"

    prediction = model.predict(X_transformed)

    if prediction[0] == 0:
        prediction = "Il s'agit d'un client sûr"

    else:
        prediction = "Il s'agit d'un client à risque"

    return {"probabilite": diff, "prediction": prediction}


# 6. Run the API
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
