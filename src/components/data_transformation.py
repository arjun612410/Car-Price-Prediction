import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np 
from src.exception import CustomException
from src.logger import logging
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            numerical_columns=['vehicle_age','km_driven','mileage','engine','max_power','seats']
            categorical_columns=['seller_type','fuel_type','transmission_type']

            num_pipeline=Pipeline(
                steps=[
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("one_hot_encoder",OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            logging.info("Numerical Columns Standard Scaling Completed")
            logging.info("Categorical Columns One Hot Encoding Completed")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            train_df=train_df.drop(columns=['car_name','brand'])
            test_df=test_df.drop(columns=['car_name','brand'])

            target_col="selling_price"

            input_feature_train_df=train_df.drop(columns=[target_col])
            target_feature_train_df=train_df[target_col]

            input_feature_test_df=test_df.drop(columns=[target_col])
            target_feature_test_df=test_df[target_col]

            preprocessing_obj=self.get_data_transformer_obj()

            logging.info("Applying Preprocessing Object on train and test")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr,target_feature_train_df]
            test_arr=np.c_[input_feature_test_arr,target_feature_test_df]

            logging.info("Saving Preprocessing Object")

            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)