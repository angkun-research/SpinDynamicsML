import pandas as pd
import numpy as np
import json

# Read the CSV file
df = pd.read_csv('./data/triangular_heisenberg_sweep.csv')

# Convert string lists to actual lists using pandas and numpy
for col in ['data', 'disp']:
    df[col] = df[col].apply(json.loads)

# Example: flatten features for regression (you may want to customize this)
X = np.array([row['data'] + row['disp'] for _, row in df.iterrows()])
y = df[['J1', 'J2', 'J3']].values

# Now you can use X and y for regression, e.g. with scikit-learn
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

# Predict example
y_pred = model.predict(X)
print(y_pred)

# total error
error = np.sum((y - y_pred) ** 2)
print(f'Total Squared Error: {error}')
