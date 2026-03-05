from flask import Flask, request, render_template
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == 'GET':
        return render_template('home.html')

    else:

        data = CustomData(

            Month=int(request.form.get("Month")),
            Age=int(request.form.get("Age")),
            Num_Bank_Accounts=int(request.form.get("Num_Bank_Accounts")),
            Num_Credit_Card=int(request.form.get("Num_Credit_Card")),
            Interest_Rate=float(request.form.get("Interest_Rate")),
            Delay_from_due_date=float(request.form.get("Delay_from_due_date")),
            Num_of_Delayed_Payment=float(request.form.get("Num_of_Delayed_Payment")),
            Changed_Credit_Limit=float(request.form.get("Changed_Credit_Limit")),
            Num_Credit_Inquiries=float(request.form.get("Num_Credit_Inquiries")),
            Credit_Utilization_Ratio=float(request.form.get("Credit_Utilization_Ratio")),
            Monthly_Balance=float(request.form.get("Monthly_Balance")),
            Credit_History_Age_months=float(request.form.get("Credit_History_Age_months")),
            Investment_Ratio=float(request.form.get("Investment_Ratio")),
            Num_Loan_Types=int(request.form.get("Num_Loan_Types")),

            Has_Mortgage=int(request.form.get("Has_Mortgage")),
            Has_Student = int(request.form.get("Has_Student")),
            Has_Personal = int(request.form.get("Has_Personal")),
            Has_Auto = int(request.form.get("Has_Auto")),
            Has_Debt_Consolidation = int(request.form.get("Has_Debt_Consolidation")),
            Has_Credit_Builder = int(request.form.get("Has_Credit_Builder")),

            Debt_to_Income=float(request.form.get("Debt_to_Income")),
            EMI_to_Income=float(request.form.get("EMI_to_Income")),
            Delay_Intensity=float(request.form.get("Delay_Intensity")),
            Utilization_Delay=float(request.form.get("Utilization_Delay")),
            Debt_per_Loan=float(request.form.get("Debt_per_Loan")),

            Occupation=request.form.get("Occupation"),
            Credit_Mix=request.form.get("Credit_Mix"),
            Payment_of_Min_Amount=request.form.get("Payment_of_Min_Amount"),
            Spending_Level=request.form.get("Spending_Level"),
            Payment_Value_Level=request.form.get("Payment_Value_Level")

        )

        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()

        results = predict_pipeline.predict(pred_df)

        return render_template("home.html", results=results[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)