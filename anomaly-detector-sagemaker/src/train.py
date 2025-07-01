import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

df = pd.read_csv('data/timeseries.csv')
model = IsolationForest(n_estimators=100, contamination=0.1)
model.fit(df[['value']])
joblib.dump(model, 'model/isolation_forest.joblib')

