Crop Yield Prediction using Linear Regression
Overview
This project predicts crop yield based on the amount of fertilizer used. The goal is to build a simple linear regression model to forecast the yield, enabling farmers and agricultural experts to optimize fertilizer usage for better crop production.

Dataset
The dataset used in this project contains two columns:

Fertilizer Used: The amount of fertilizer applied to the crops.
Crop Yield: The resulting crop yield.
Approach
Data Preprocessing:

The dataset is loaded from a CSV file (data.csv).
The data is split into features (fertilizer used) and target (crop yield).
Model Training:

A Linear Regression model is trained using the training set.
The model is tested using a separate test set.
Model Evaluation:

The model's performance is evaluated using two metrics:
R-squared (R²): A measure of how well the model explains the variance in the data. A high R² indicates a good fit.
Mean Squared Error (MSE): A metric that measures the average of the squared errors, with a lower value indicating better accuracy.
Visualization:

The relationship between fertilizer usage and crop yield is visualized using scatter plots for both the training and test sets.
Performance
R-squared: 0.9927, indicating that the model explains 99.27% of the variance in the crop yield based on the amount of fertilizer used.
Mean Squared Error (MSE): 0.0762, indicating low prediction error.
Conclusion
The model is highly accurate, providing a reliable prediction of crop yield based on fertilizer usage, with a high R² score and low MSE. This can be useful for optimizing fertilizer usage in agricultural practices.

Requirements
Python 3.x
Libraries:
pandas
numpy
matplotlib
scikit-learn
How to Run
Clone this repository to your local machine.
Install the required libraries:
bash
Copy code
pip install -r requirements.txt
Run the script to train the model and visualize the results:
bash
Copy code
python train_model.py
Future Improvements
Experiment with more features (e.g., temperature, rainfall) to improve model accuracy.
Explore more advanced regression techniques like polynomial regression or multiple linear regression.
