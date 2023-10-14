import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the California housing dataset
housing = fetch_california_housing()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.2, random_state=42)

# Create and train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate and print the mean absolute error and mean squared error
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print('Mean Absolute Error =', mae)
print('Mean Squared Error =', mse)

# Create a new data point for prediction
new_data = np.array([[9.12, 38.5, 5.0, 1.02, 678.0, 3.0, 38.0, -123.0]])

# Reshape the new data point (optional if you want to keep it as 2D array)
new_data = new_data.reshape(1, -1)

# Use the trained model to predict the price of the new data point
predicting_price = model.predict(new_data)
print('Predicted Price for New Data =', predicting_price)
