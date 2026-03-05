import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, features):

        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"
            label_encoder_path = "artifacts/label_encoder.pkl"

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            label_encoder = load_object(label_encoder_path)

            data_scaled = preprocessor.transform(features)

            pred_numeric = model.predict(data_scaled)

            prediction = label_encoder.inverse_transform(pred_numeric)

            return prediction

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:

    def __init__(
        self,
        Month,
        Age,
        Num_Bank_Accounts,
        Num_Credit_Card,
        Interest_Rate,
        Delay_from_due_date,
        Num_of_Delayed_Payment,
        Changed_Credit_Limit,
        Num_Credit_Inquiries,
        Credit_Utilization_Ratio,
        Monthly_Balance,
        Credit_History_Age_months,
        Investment_Ratio,
        Num_Loan_Types,
        Has_Mortgage,
        Has_Student,
        Has_Personal,
        Has_Auto,
        Has_Debt_Consolidation,
        Has_Credit_Builder,
        Debt_to_Income,
        EMI_to_Income,
        Delay_Intensity,
        Utilization_Delay,
        Debt_per_Loan,
        Occupation,
        Credit_Mix,
        Payment_of_Min_Amount,
        Spending_Level,
        Payment_Value_Level
    ):

        self.Month = Month
        self.Age = Age
        self.Num_Bank_Accounts = Num_Bank_Accounts
        self.Num_Credit_Card = Num_Credit_Card
        self.Interest_Rate = Interest_Rate
        self.Delay_from_due_date = Delay_from_due_date
        self.Num_of_Delayed_Payment = Num_of_Delayed_Payment
        self.Changed_Credit_Limit = Changed_Credit_Limit
        self.Num_Credit_Inquiries = Num_Credit_Inquiries
        self.Credit_Utilization_Ratio = Credit_Utilization_Ratio
        self.Monthly_Balance = Monthly_Balance
        self.Credit_History_Age_months = Credit_History_Age_months
        self.Investment_Ratio = Investment_Ratio
        self.Num_Loan_Types = Num_Loan_Types
        self.Has_Mortgage = Has_Mortgage
        self.Has_Student = Has_Student
        self.Has_Personal = Has_Personal
        self.Has_Auto = Has_Auto
        self.Has_Debt_Consolidation = Has_Debt_Consolidation
        self.Has_Credit_Builder = Has_Credit_Builder
        self.Debt_to_Income = Debt_to_Income
        self.EMI_to_Income = EMI_to_Income
        self.Delay_Intensity = Delay_Intensity
        self.Utilization_Delay = Utilization_Delay
        self.Debt_per_Loan = Debt_per_Loan
        self.Occupation = Occupation
        self.Credit_Mix = Credit_Mix
        self.Payment_of_Min_Amount = Payment_of_Min_Amount
        self.Spending_Level = Spending_Level
        self.Payment_Value_Level = Payment_Value_Level


    def get_data_as_data_frame(self):

        try:

            custom_data_input_dict = {

                "Month": [self.Month],
                "Age": [self.Age],
                "Num_Bank_Accounts": [self.Num_Bank_Accounts],
                "Num_Credit_Card": [self.Num_Credit_Card],
                "Interest_Rate": [self.Interest_Rate],
                "Delay_from_due_date": [self.Delay_from_due_date],
                "Num_of_Delayed_Payment": [self.Num_of_Delayed_Payment],
                "Changed_Credit_Limit": [self.Changed_Credit_Limit],
                "Num_Credit_Inquiries": [self.Num_Credit_Inquiries],
                "Credit_Utilization_Ratio": [self.Credit_Utilization_Ratio],
                "Monthly_Balance": [self.Monthly_Balance],
                "Credit_History_Age(months)": [self.Credit_History_Age_months],
                "Investment_Ratio": [self.Investment_Ratio],
                "Num_Loan_Types": [self.Num_Loan_Types],
                "Has_Mortgage": [self.Has_Mortgage],
                "Has_Student": [self.Has_Student],
                "Has_Personal": [self.Has_Personal],
                "Has_Auto": [self.Has_Auto],
                "Has_Debt_Consolidation": [self.Has_Debt_Consolidation],
                "Has_Credit_Builder": [self.Has_Credit_Builder],
                "Debt_to_Income": [self.Debt_to_Income],
                "EMI_to_Income": [self.EMI_to_Income],
                "Delay_Intensity": [self.Delay_Intensity],
                "Utilization_Delay": [self.Utilization_Delay],
                "Debt_per_Loan": [self.Debt_per_Loan],
                "Occupation": [self.Occupation],
                "Credit_Mix": [self.Credit_Mix],
                "Payment_of_Min_Amount": [self.Payment_of_Min_Amount],
                "Spending_Level": [self.Spending_Level],
                "Payment_Value_Level": [self.Payment_Value_Level]

            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)