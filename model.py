import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

print("🚀 Loading dataset...")

df = pd.read_csv("rainfall_predictions.csv")
df.columns = df.columns.str.strip()
df["DISTRICT_CODE"] = df["DISTRICT"].astype("category").cat.codes

# Safety check for required column
if "DISTRICT" not in df.columns:
    raise ValueError("DISTRICT column missing in dataset")

# ===================== FEATURES =====================
X = df[['JUN', 'JUL', 'AUG', 'SEP', 'DISTRICT_CODE']]

# ===================== TARGET =====================
y = df['ANNUAL']
# ===================== SPLIT =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===================== MODELS =====================
models = {
    "Decision Tree": DecisionTreeRegressor(),
    "KNN": KNeighborsRegressor(),
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}

# ===================== TRAIN & EVALUATE =====================
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    error = mean_absolute_error(y_test, preds)
    results[name] = error

# ===================== BEST MODEL =====================
best_model_name = min(results, key=results.get)
best_model = models[best_model_name]

print("Model Errors:", results)
print("Best Model:", best_model_name)

# ===================== FINAL TRAIN =====================
best_model.fit(X, y)

df["Prediction"] = best_model.predict(X)

# Save
df.to_csv("final_dataset.csv", index=False)

print("✅ Model improved and saved!")
import joblib
joblib.dump(best_model, "model.pkl")