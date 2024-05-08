import os
import sys
from sklearn.ensemble import (
        AdaBoostRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor
    )
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from source.exception import CustomException
from source.logger import logging
from source.utils import save_object

from dataclasses import dataclass
from source.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info('splitting training and testing data')
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models={
                "Random Forest": RandomForestRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "XGB Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "KNeighbors Regressor": KNeighborsRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor()
            }

            params={
                "Random Forest":{
                    'n_estimators':[50,80,100,150],
                    'criterion':['absolute_error','squared_error','poisson','friedman_mse'],
                    'max_depth':[3,5,8,10],
                    'min_samples_split':[2,3,4,5]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[0.001,0.01, 0.1, 0.2, 0.5,1,2],
                    'loss':['linear','square','exponential']
                },
                "XGB Regressor":{
                    'lambda':[0.1,0.2,0.5,1,2,5,10],
                    'alpha':[0.1,0.2,0.5,1,2,5,10],
                    'learning_rate' : [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
                },
                "CatBoost Regressor":{
                    'depth'         : [6,8,10],
                  'learning_rate' : [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
                  'iterations'    : [30, 50, 100,150]
                },
                "Gradient Boosting Regressor":{
                    #'loss':['squared_error','absolute_error','huber','quantile'],
                    'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
                    'n_estimators':[50,80,100,150],
                    #'criterion':['friedman_mse','squared_error'],
                    'min_samples_split':[2,3,4,5],
                    'max_depth':[3,5,8,10],
                    'alpha':[0.5,0.7,0.9,0.98]
                },
                "Linear Regression":{
                },
                "KNeighbors Regressor":{
                    'n_neighbors':[3,5,6,7,9,10],
                    'weights':['uniform','distance']
                },
                "Decision Tree Regressor":{
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    'splitter':['best','random'],
                    'max_depth':[3,5,7,9],
                    'min_samples_split':[2,3,4,5]
                }
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                              models=models,params=params)

            #to get the best model score from the dictionary
            best_model_score=max(sorted(model_report.values()))

            #to get best model name
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            if(best_model_score<0.6):
                raise CustomException("no best model found")
            
            logging.info("best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2Score=r2_score(y_test,predicted)
            return r2Score

        except Exception as e:
            raise CustomException(e,sys)
