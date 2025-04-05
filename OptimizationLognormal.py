from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from ngboost import NGBRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from ngboost.distns import LogNormal, Normal
from sklearn.base import clone
import numpy as np



# Load the data
path = "Dataset.csv"
blasting_data = pd.read_csv(path)

# Choose one target variable
X = blasting_data.drop(columns=['Maximum displacement(mm)'])
Y = blasting_data['Maximum displacement(mm)']

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


class CustomDecisionTreeRegressor(DecisionTreeRegressor):
    def __init__(self, criterion='friedman_mse', max_depth=None, min_samples_split=2, min_samples_leaf=1):
        super().__init__(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )

# Define the parameter grid for GridSearchCV

param_grid = {
    'Base__max_depth':  [6,8,10,12,14]],
    'Base__min_samples_split': [2,3,4,5,6,7],
    'Base__min_samples_leaf': [1,2,4,6,8,10],
    'n_estimators': [800, 1000,1200,1400,1600],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2]
}



ngb = NGBRegressor(Dist=LogNormal, verbose=False, Base=CustomDecisionTreeRegressor())

# Set up GridSearchCV
grid_search = GridSearchCV(ngb, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1)
grid_search.fit(X, Y)

# Print the best parameters
print(grid_search.best_params_)