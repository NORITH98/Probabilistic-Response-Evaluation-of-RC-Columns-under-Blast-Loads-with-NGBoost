from ngboost import NGBRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from ngboost.distns import LogNormal, Normal
import numpy as np
import shap

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

# Instantiate and fit the NGBRegressor with Normal distribution
base_regressor = CustomDecisionTreeRegressor(max_depth=4, min_samples_leaf=3, min_samples_split=4)
ngbNormal = NGBRegressor(Dist=Normal, verbose=False, Base=base_regressor, learning_rate=0.05, n_estimators=460).fit(X_train_scaled, Y_train)

# SHAP analysis
shap.initjs()
explainer = shap.TreeExplainer(ngbNormal, model_output=0) 
shap_values_normalLoc = explainer.shap_values(X)
shap.summary_plot(shap_values=shap_values_normalLoc, features=X, feature_names=['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13'], show=False)


# SHAP analysis
shap.initjs()
explainer = shap.TreeExplainer(ngbNormal, model_output=1) 
shap_values_normalScale = explainer.shap_values(X)
shap.summary_plot(shap_values=shap_values_normalScale, features=X, feature_names=['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13'], show=False)
