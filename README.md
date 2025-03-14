# workshop-ml
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset
file_path = "/content/FuelConsumption.csv"
df = pd.read_csv(file_path)

# Scatter Plot 1: CYLINDERS vs CO2EMISSIONS (green color)
plt.figure(figsize=(6, 4))
plt.scatter(df["CYLINDERS"], df["CO2EMISSIONS"], color='green', alpha=0.5)
plt.xlabel("Cylinders")
plt.ylabel("CO2 Emissions (g/km)")
plt.title("Cylinders vs CO2 Emissions")
plt.grid(True)
plt.show()

# Scatter Plot 2: Compare CYLINDERS vs CO2EMISSIONS and ENGINESIZE vs CO2EMISSIONS
plt.figure(figsize=(6, 4))
plt.scatter(df["CYLINDERS"], df["CO2EMISSIONS"], color='blue', alpha=0.5, label="Cylinders vs CO2 Emissions")
plt.scatter(df["ENGINESIZE"], df["CO2EMISSIONS"], color='red', alpha=0.5, label="Engine Size vs CO2 Emissions")
plt.xlabel("Cylinders / Engine Size")
plt.ylabel("CO2 Emissions (g/km)")
plt.title("Comparison of Cylinders & Engine Size vs CO2 Emissions")
plt.legend()
plt.grid(True)
plt.show()

# Scatter Plot 3: Compare CYLINDERS, ENGINESIZE, and FUELCONSUMPTION_COMB vs CO2EMISSIONS
plt.figure(figsize=(6, 4))
plt.scatter(df["CYLINDERS"], df["CO2EMISSIONS"], color='blue', alpha=0.5, label="Cylinders vs CO2 Emissions")
plt.scatter(df["ENGINESIZE"], df["CO2EMISSIONS"], color='red', alpha=0.5, label="Engine Size vs CO2 Emissions")
plt.scatter(df["FUELCONSUMPTION_COMB"], df["CO2EMISSIONS"], color='purple', alpha=0.5, label="Fuel Consumption vs CO2 Emissions")
plt.xlabel("Cylinders / Engine Size / Fuel Consumption")
plt.ylabel("CO2 Emissions (g/km)")
plt.title("Comparison of Cylinders, Engine Size & Fuel Consumption vs CO2 Emissions")
plt.legend()
plt.grid(True)
plt.show()

# Train Model 1: Cylinders as independent variable
X_cylinders = df[["CYLINDERS"]]
y_co2 = df["CO2EMISSIONS"]
X_train, X_test, y_train, y_test = train_test_split(X_cylinders, y_co2, test_size=0.2, random_state=42)
model_cylinders = LinearRegression()
model_cylinders.fit(X_train, y_train)
y_pred = model_cylinders.predict(X_test)
r2_cylinders = r2_score(y_test, y_pred)
print(f"Model Accuracy (Cylinders as predictor): {r2_cylinders:.4f}")

# Train Model 2: Fuel Consumption as independent variable
X_fuel = df[["FUELCONSUMPTION_COMB"]]
X_train, X_test, y_train, y_test = train_test_split(X_fuel, y_co2, test_size=0.2, random_state=42)
model_fuel = LinearRegression()
model_fuel.fit(X_train, y_train)
y_pred_fuel = model_fuel.predict(X_test)
r2_fuel = r2_score(y_test, y_pred_fuel)
print(f"Model Accuracy (Fuel Consumption as predictor): {r2_fuel:.4f}")

# Train models with different train-test ratios and store accuracies
ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
accuracy_results = {}

for ratio in ratios:
    X_train, X_test, y_train, y_test = train_test_split(X_fuel, y_co2, test_size=ratio, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    accuracy_results[f"Train {int((1-ratio)*100)}% - Test {int(ratio*100)}%"] = r2

# Print accuracy results for different train-test splits
for key, value in accuracy_results.items():
    print(f"{key}: R^2 Score = {value:.4f}")

```
### output:

<img width="662" alt="422633030-aa5a02b4-a991-4931-93d7-691c223210ee" src="https://github.com/user-attachments/assets/a86f4790-eb14-4ee2-888c-52c3e67d3796" />





<img width="436" alt="422632945-4086edc2-db43-41de-89bb-97bb7f6ea34f" src="https://github.com/user-attachments/assets/67be006b-edd0-4e76-9d1d-9e01134c9f59" />








<img width="532" alt="422633317-744d44f3-a6c1-458f-aadc-bfeea2213d6d" src="https://github.com/user-attachments/assets/7e43143a-ba22-4095-86d0-9daeb689c1d4" />
