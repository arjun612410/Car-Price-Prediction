import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_obj

class PredictPipelie:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path="artifacts/model.pkl"
            preprocessor_path="artifacts/preprocessor.pkl"
            model=load_obj(file_path=model_path)
            preprocessor=load_obj(file_path=preprocessor_path)
            features=features.fillna("Missing")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
            car_name:str,
            brand:str,
            model:str,
            vechile_age:int,
            km_driven:int,
            seller_type:str,
            fuel_type:str,
            transmission_type:str,
            mileage:float,
            engine:int,
            max_power:float,
            seats:int):
        self.car_name=car_name
        self.brand=brand
        self.model=model
        self.vechile_age=vechile_age
        self.km_driven=km_driven
        self.seller_type=seller_type
        self.fuel_type=fuel_type
        self.transmssion_type=transmission_type
        self.mileage=mileage
        self.engine=engine
        self.max_power=max_power
        self.seats=seats

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict={
                "car_name":[self.car_name],
                "brand":[self.brand],
                "model":[self.model],
                "vechile_age":[self.vechile_age],
                "km_driven":[self.km_driven],
                "seller_type":[self.seller_type],
                "fuel_type":[self.fuel_type],
                "transmission_type":[self.transmssion_type],
                "mileage":[self.mileage],
                "engine":[self.engine],
                "max_power":[self.max_power],
                "seats":[self.seats]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)