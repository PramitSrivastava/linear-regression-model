from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset from CSV
data = pd.read_csv('data.csv')
X = data.iloc[:, :-1].values  # Fertilizer (single feature)
y = data.iloc[:, -1].values   # Crop Yield (target)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/3, random_state=0)

# Create and train the linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict on the test set
y_pred = regressor.predict(X_test)

# Calculate R-squared and Mean Squared Error
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R-squared: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")

Plotting the results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Crop Yield vs Fertilizer (Training set)')
plt.xlabel('Fertilizer Used')
plt.ylabel('Crop Yield')
plt.show()

plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Crop Yield vs Fertilizer (Test set)')
plt.xlabel('Fertilizer Used')
plt.ylabel('Crop Yield')
plt.show()

# Print final message
print("Model trained and evaluated successfully!")
