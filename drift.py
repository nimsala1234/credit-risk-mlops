import sys
import joblib
import pandas as pd

bundle = joblib.load("artifacts/model.joblib")
batch = pd.read_csv(sys.argv[1])
cols = bundle["columns"]
mean = pd.Series(bundle["mean"])
std = pd.Series(bundle["std"]).replace(0, float("nan"))
shift = ((batch[cols].mean() - mean) / std).abs()
print(shift.sort_values(ascending=False))
print("Features with mean shift > 0.5 standard deviations:")
print(shift[shift > .5])
