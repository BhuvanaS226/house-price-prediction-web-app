import pandas as pd
import numpy as np
import pickle


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# 1️⃣ Load Dataset
housing = fetch_california_housing()

# Convert to DataFrame
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["Price"] = housing.target


# 2️⃣ Define Features (X) and Target (y)
X = df.drop("Price", axis=1)
y = df["Price"]


# 3️⃣ Split Data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4️⃣ Create and Train Model
model = LinearRegression()
model.fit(X_train, y_train)


# 5️⃣ Make Predictions on Test Data
y_pred = model.predict(X_test)


# 6️⃣ Evaluate Model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("📊 Model Evaluation Results")
print("----------------------------")
print("RMSE:", rmse)
print("R² Score:", r2)


# 7️⃣ Save the Trained Model
pickle.dump(model, open("house_model.pkl", "wb"))

print("\n✅ Model saved as house_model.pkl")