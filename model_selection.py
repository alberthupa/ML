# quickest
https://github.com/shankarpandala/lazypredict

# REGRESSION

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import numpy as np

# Initialize the models
models = {
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "ElasticNet": ElasticNet(),
    "DecisionTree": DecisionTreeRegressor(),
    "RandomForest": RandomForestRegressor(),
    "GradientBoosting": GradientBoostingRegressor(),
    "SupportVector": SVR(),
    "XGBoost": XGBRegressor(),
    "LightGBM": LGBMRegressor()
}

# Train and evaluate each model
rmse_scores = {}

for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    # Store the RMSE
    rmse_scores[model_name] = rmse

rmse_scores


# BINARY CLASSIFICATION

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np

# Initialize the models
models = {
    "Logistic": LogisticRegression(),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "SupportVector": SVC(),
    "XGBoost": XGBClassifier(),
    "LightGBM": LGBMClassifier()
}

# Train and evaluate each model
accuracy_scores = {}

for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Compute Accuracy
    accuracy = accuracy_score(y_test, predictions)
    
    # Store the accuracy score
    accuracy_scores[model_name] = accuracy

accuracy_scores



