import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_absolute_error

# Load data
df = pd.read_csv("rainfall.csv")

# Features and target
X = df[['Actual Rainfall: JUN',
        'Actual Rainfall: JUL',
        'Actual Rainfall: AUG',
        'Actual Rainfall: SEPT']]

y = df['Actual Rainfall: JUN-SEPT']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Decision Tree
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
df['DT_Prediction'] = dt.predict(X)

# KNN
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y_train)
df['KNN_Prediction'] = knn.predict(X)

# Naive Bayes (classification)
df['Rainfall_Class'] = (y > y.mean()).astype(int)
nb = GaussianNB()
nb.fit(X, df['Rainfall_Class'])
df['NB_Prediction'] = nb.predict(X)

# Evaluation
print("DT Error:", mean_absolute_error(y, df['DT_Prediction']))
print("KNN Error:", mean_absolute_error(y, df['KNN_Prediction']))

# Save
df.to_csv("rainfall_predictions.csv", index=False)

print("✅ Model completed and file saved!")


