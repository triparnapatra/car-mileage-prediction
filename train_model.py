
import joblib
import pandas as pd

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the Auto MPG dataset
dataset = fetch_ucirepo(id=9)

X = dataset.data.features.copy()
y = dataset.data.targets["mpg"]

# 2. Remove car names and convert columns to numbers
X = X.drop(columns=["car_name"], errors="ignore")
X = X.apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(y, errors="coerce")

# 3. Remove rows with missing values
data = X.copy()
data["mpg"] = y
data = data.dropna()

X = data.drop(columns=["mpg"])
y = data["mpg"]

# 4. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train the model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# 6. Evaluate the model
predictions = model.predict(X_test)

print("MAE:", round(mean_absolute_error(y_test, predictions), 2))
print("R2 Score:", round(r2_score(y_test, predictions), 3))

# 7. Save the model and feature names
joblib.dump(
    {"model": model, "features": list(X.columns)},
    "car_mileage_model.pkl"
)

print("Model saved successfully!")
