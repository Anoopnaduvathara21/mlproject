import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()                  

        """
        Method 1

        This builds the preprocessing pipelines.
        
        """

    def get_data_transformer_object(self):

        """
        this function is reponsible for data transformation

        """
        try:
            numeric_features = ["writing_score","reading_score"]     #Splits columns into numeric and categorical.
            categorical_features = [
                "gender",
                "race_ethnicity", 
                "parental_level_of_education", 
                "lunch", 
                "test_preparation_course",
            ]
        

            num_pipeline = Pipeline(                                #Numeric pipeline = fill missing values with median, then scale.
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(                      #Categorical pipeline = fill missing with most frequent, encode categories, scale.
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical columns standard scaling completed")
            logging.info("categorical columns encoding completed")

            preprocessor=ColumnTransformer(                      #Combines both pipelines into a single preprocessing object.
                [
                ("num_pipeline",num_pipeline,numeric_features),
                ("cat_pipeline",cat_pipeline,categorical_features)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):      #method 2
        try:
            train_df = pd.read_csv(train_path)               #Reads train/test CSVs.
            test_df = pd.read_csv(test_path)

            logging.info("read train and test data completed")
            logging.info("obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"                                   #Splits input features vs target.
            numeric_features = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df) #Fits preprocessing on train, applies to both train and test.
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]#Concatenates processed features with target into arrays.
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Attempting to save preprocessing object to {self.data_transformation_config.preprocessor_obj_file_path}")
            try:
                save_object(
                    file_path=self.data_transformation_config.preprocessor_obj_file_path,#Saves the preprocessing object as preprocessor.pkl for reuse.
                    obj=preprocessing_obj
                )
                logging.info(f"Successfully saved preprocessing object to {self.data_transformation_config.preprocessor_obj_file_path}")
            except Exception as save_err:
                logging.error(f"Failed to save preprocessing object: {save_err}")
                raise

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            logging.error(f"Exception in initiate_data_transformation: {e}")
            raise CustomException(e, sys)