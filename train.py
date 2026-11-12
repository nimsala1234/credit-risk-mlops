from pathlib import Path
import sqlite3
import json
import joblib
import pandas as pd
import mlflow
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss
)

def metrics(y, p):
    return {
        "roc_auc": float(roc_auc_score(y,p)),
        "average_precision": float(average_precision_score(y,p)),
        "brier_score": float(brier_score_loss(y,p))
    }

def main():
    dataset = fetch_ucirepo(id=350)
    X = dataset.data.features.copy()
    y = dataset.data.targets.iloc[:,0].astype(int)
    # Exclude sensitive demographic fields in this baseline.
    X = X.drop(columns=["X2","X3","X4","X5","SEX",
        "EDUCATION","MARRIAGE","AGE"], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")
    with sqlite3.connect("credit.db") as db:
        X.assign(target=y.to_numpy()).to_sql(
            "credit_records", db, if_exists="replace", index=False
        )
        data = pd.read_sql_query("SELECT * FROM credit_records", db)
    y = data.pop("target")
    X = data
    Xt, Xtest, yt, ytest = train_test_split(
        X,y,test_size=.2,stratify=y,random_state=42
    )
    Xtrain,Xval,ytrain,yval = train_test_split(
        Xt,yt,test_size=.25,stratify=yt,random_state=42
    )

    models = {
        "logistic_regression": make_pipeline(
            SimpleImputer(), StandardScaler(),
            LogisticRegression(max_iter=1000)
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            random_state=42
        )
    }
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("credit-default")
    best, best_score = None, -1
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(Xtrain,ytrain)
            result = metrics(yval,model.predict_proba(Xval)[:,1])
            mlflow.log_metrics(result)
            print(name,result)
            if result["average_precision"] > best_score:
                best, best_score = model,result["average_precision"]

    report = metrics(ytest,best.predict_proba(Xtest)[:,1])
    Path("artifacts").mkdir(exist_ok=True)
    joblib.dump({
        "model":best, "columns":list(X.columns),
        "mean":Xtrain.mean().to_dict(),
        "std":Xtrain.std().to_dict()
    }, "artifacts/model.joblib")
    Path("artifacts/metrics.json").write_text(json.dumps(report,indent=2))
    with mlflow.start_run(run_name="held-out-evaluation"):
        mlflow.log_metrics(report)
        mlflow.log_artifact("artifacts/model.joblib")
    print("Held-out evaluation:",report)

if __name__ == "__main__":
    main()
