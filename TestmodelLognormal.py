from ngboost.distns import LogNormal, Normal
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ngboost import NGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from matplotlib import font_manager as fm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# Load the data
path = "Dataset.csv"
blasting_data = pd.read_csv(path)

# Choose one target variable
X = blasting_data.drop(columns=['Maximum displacement(mm)'])
Y = blasting_data['Maximum displacement(mm)']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# Standardize input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled_full = scaler.transform(X)  


# Define the custom decision tree regressor
class CustomDecisionTreeRegressor(DecisionTreeRegressor):
    def __init__(self, criterion='friedman_mse', max_depth=None, min_samples_split=2, min_samples_leaf=1):
        super().__init__(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

base_regressor = CustomDecisionTreeRegressor(max_depth=10, min_samples_leaf=4, min_samples_split=6)
ngbLogNormal = NGBRegressor(Dist=LogNormal, verbose=False, Base=base_regressor, learning_rate=0.1, n_estimators=1400).fit(X_train_scaled, Y_train)

# Print the predicted and actual value for the traing set
y_predictionLognormal=ngbLogNormal.pred_dist(X_scaled_full).params['scale']
r2 = r2_score(Y, y_predictionLognormal)
rmse = mean_squared_error(Y, y_predictionLognormal, squared=False)

# Print the results
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)