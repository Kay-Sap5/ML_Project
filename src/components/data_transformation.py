import sys
from dataclasses import dataclass

import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path= os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    
    def get_data_transformation_obj(self):

        """
        This function is responsible for data Transformation 
        """


        try:
            num_columns = ['reading_score', 'writing_score']
            cat_columns = ['gender', 
                           'race_ethnicity',
                             'parental_level_of_education', 
                             'lunch', 
                             'test_preparation_course']
            
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ('ohe',OneHotEncoder()),
                   
                ]
            )

            logging.info(f"Categorical Columns: {cat_columns}")
            logging.info(f"Numerical Columns : {num_columns}")


            preprocessor = ColumnTransformer(
                [
                    ("cat_pipleine",cat_pipeline,cat_columns),
                    ('num_pipeline',num_pipeline,num_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_data_path , test_data_path):
        try:
           
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Read Train and Test Data Complete....")

            logging.info("Obtaining preprocessor object")
            
            preprocessor = self.get_data_transformation_obj()

            target_column = 'math_score'

            input_feature_train_df = train_df.drop(columns=[target_column],axis = 1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns =[target_column],axis = 1)
            target_feature_test_df = test_df[target_column]

            logging.info("Applying Preprocessing object on training and testing dataframe")

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr  = preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr , np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr , np.array(target_feature_test_df)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor
            )

            logging.info(f"Saved Processing obj")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)
            