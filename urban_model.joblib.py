import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# 1. Generate Synthetic Urban Data (10,000 records)
np.random.seed(42)
n_samples = 10000

data = {
    'ndvi': np.random.uniform(0, 1, n_samples),          # Vegetation
    'albedo': np.random.uniform(0.1, 0.9, n_samples),    # Reflectivity
    'density': np.random.uniform(0, 100, n_samples),     # Building Density (%)
    'svf': np.random.uniform(0, 1, n_samples)            # Sky View Factor
}

df = pd.DataFrame(data)

# 2. Create a target: Predicted Surface Temperature
# Higher density = Higher temp | Higher NDVI/Albedo = Lower temp
df['temperature'] = (25 + (df['density'] * 0.15) - (df['ndvi'] * 5) - (df['albedo'] * 4) + np.random.normal(0, 1, n_samples))

# 3. Train the Random Forest Regressor
X = df[['ndvi', 'albedo', 'density', 'svf']]
y = df['temperature']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 4. Save the model as a file
joblib.dump(model, 'urban_model.joblib')

print("Success! 'urban_model.joblib' has been created.")