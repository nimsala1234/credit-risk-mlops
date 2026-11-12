import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Credit Default Risk API")

class Record(BaseModel):
    features: dict[str, float]

@app.get("/health")
def health():
    return {"status":"ok","model_exists":os.path.isfile(
        "artifacts/model.joblib"
    )}

@app.post("/predict")
def predict(record: Record):
    path = "artifacts/model.joblib"
    if not os.path.isfile(path):
        raise HTTPException(503,"Train the model first.")
    bundle = joblib.load(path)
    if set(record.features) != set(bundle["columns"]):
        raise HTTPException(422,detail={
            "required_features":bundle["columns"]
        })
    frame = pd.DataFrame([record.features])[bundle["columns"]]
    probability = bundle["model"].predict_proba(frame)[0,1]
    return {"default_probability":float(probability)}
