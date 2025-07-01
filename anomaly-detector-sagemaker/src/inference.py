import pandas as pd
import joblib

df = pd.read_csv('data/timeseries.csv')
model = joblib.load('model/isolation_forest.joblib')
df['anomaly'] = model.predict(df[['value']])
df.to_csv('data/predicted.csv', index=False)

