import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, LabelEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(PROJECT_ROOT, "artifacts", "preprocessor.pkl")
    target_encoder_file_path = os.path.join(PROJECT_ROOT, "artifacts", "label_encoder.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:

            numerical_columns = [
                'Month','Age','Num_Bank_Accounts','Num_Credit_Card','Interest_Rate',
                'Delay_from_due_date','Num_of_Delayed_Payment','Changed_Credit_Limit',
                'Num_Credit_Inquiries','Credit_Utilization_Ratio','Monthly_Balance',
                'Credit_History_Age(months)','Investment_Ratio','Num_Loan_Types',
                'Has_Mortgage','Has_Student','Has_Personal','Has_Auto',
                'Has_Debt_Consolidation','Has_Credit_Builder','Debt_to_Income',
                'EMI_to_Income','Delay_Intensity','Utilization_Delay','Debt_per_Loan'
            ]

            OHE_categorical_columns = ['Occupation']

            OE_categorical_columns = [
                'Credit_Mix',
                'Payment_of_Min_Amount',
                'Spending_Level',
                'Payment_Value_Level'
            ]

            categories = [
                ['Bad', 'Standard', 'Good'],
                ['No', 'Yes'],
                ['Low', 'High'],
                ['Small', 'Medium', 'Large']
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            ohe_cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            oe_cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ordinal", OrdinalEncoder(categories=categories))
                ]
            )

            logging.info("Creating preprocessing pipelines")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("oe_cat_pipeline", oe_cat_pipeline, OE_categorical_columns),
                    ("ohe_cat_pipeline", ohe_cat_pipeline, OHE_categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded")

            preprocessing_obj = self.get_data_transformer_object()

            target_column = "Credit_Score"

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            logging.info("Applying preprocessing")

            X_train = preprocessing_obj.fit_transform(X_train)
            X_test = preprocessing_obj.transform(X_test)

            label_encoder = LabelEncoder()

            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)

            logging.info("Saving preprocessing objects")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            save_object(
                file_path=self.data_transformation_config.target_encoder_file_path,
                obj=label_encoder
            )

            logging.info("Data transformation completed")

            return (
                X_train,
                y_train,
                X_test,
                y_test,
                self.data_transformation_config.preprocessor_obj_file_path,
                self.data_transformation_config.target_encoder_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)