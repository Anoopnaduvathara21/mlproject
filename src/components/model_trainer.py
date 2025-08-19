import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass                                  #Simple dataclass holding where the trained model will be saved
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:                       #Instantiate config when the trainer is created
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):           #initiate_model_trainer(self, train_array, test_array)

                                                
        try:
            logging.info("spliting training and test input data")    # 1) Split features/target from the arrays
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]                                      #Assumes last column is the target; everything before it is features

            )
            models = {                                                #2) Define candidate models
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),                        #A registry of estimators to try
            }

            params = {                                          #3) Define hyperparameter grids
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "K-Neighbors Regressor": {
                    'n_neighbors': [5, 7, 9, 11],
                },
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }                                                      #Search space for each model (empty dict = use defaults)


            model_report:dict = evaluate_models(X_train = X_train, y_train = y_train,       #4) Evaluate all models
                                               X_test = X_test, y_test = y_test, models=models,param = params)
            
            ## to get bst model from the dict                         
            best_model_score = max(sorted(model_report.values()))    
                                                                   
            """
            Expects a dict like {model_name: score} after tuning/evaluation


(Important: your evaluate_models must fit models—often via GridSearchCV—and either return scores and keep fitted estimators in-place,
 or return the best estimator. This script assumes the objects in models end up fitted.)
 """                                                     
            ##to get the best model name from the dict 5) Pick the best model by score
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model =models[best_model_name]        #Finds the top score and retrieves the corresponding estimator from models

            if best_model_score<0.6:                          #6) Guardrail: require minimum quality 
                raise CustomException("no best mode found")   #Fails fast if R² (presumably) is too low
            
            logging.info(f"best found model on both training and testing dataset")

            save_object(                         #7) Persist the model
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )                         #Dumps the (fitted) estimator to artifacts/model.pkl

            predicted = best_model.predict(X_test)         #8) Final metric

            r2_square = r2_score(y_test, predicted)
            return r2_square          #Computes R² on the test set and returns it

             
        except Exception as e:
            raise CustomException(e,sys)   
        """Micro-notes to keep in mind

The code relies on evaluate_models to fit models (and, by side effect, the chosen best_model). 
If evaluate_models returns only scores but doesnt fit/assign back, best_model.predict(...) would error. 
Make sure your evaluate_models fits and leaves the best estimator ready to use (or returns it).

The selection uses the max score from model_report; ensure all scores are comparable (same metric and direction).
"""
