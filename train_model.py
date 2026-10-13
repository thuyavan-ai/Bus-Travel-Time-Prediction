# Bus Travel Time Prediction using Linear Regression
# Beginner-friendly Machine Learning project

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the dataset
data = pd.read_csv("bus_dataset.csv")
print("Dataset:")
print(data)

# 2. Select input (X) and output (y)
X = data[["Distance_km"]]
y = data["Travel_Time_minutes"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Test the trained model
predictions = model.predict(X_test)
print("\nActual travel times:", list(y_test))
print("Predicted travel times:", [round(value, 2) for value in predictions])
print("Mean Absolute Error:", round(mean_absolute_error(y_test, predictions), 2))
print("R-squared score:", round(r2_score(y_test, predictions), 2))

# 6. Predict travel time for a new bus journey
distance = float(input("\nEnter bus journey distance in km: "))
new_data = pd.DataFrame({"Distance_km": [distance]})
predicted_time = model.predict(new_data)[0]
print(f"Estimated travel time for {distance:g} km: {predicted_time:.2f} minutes")

# 7. Plot the dataset and regression line
plt.scatter(data["Distance_km"], data["Travel_Time_minutes"], label="Dataset")
plt.plot(
    data["Distance_km"],
    model.predict(data[["Distance_km"]]),
    label="Regression line"
)
plt.xlabel("Distance (km)")
plt.ylabel("Travel time (minutes)")
plt.title("Bus Travel Time Prediction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("regression_plot.png")
plt.show()
