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

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

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
