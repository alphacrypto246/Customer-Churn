from fastapi import FastAPI
from pydantic import BaseModel

from src.pipeline.predict_pipeline import PredictPipeline


app = FastAPI(title="Customer Churn Prediction API")


class CustomerData(BaseModel):

    AccountWeeks: int
    ContractRenewal: int
    DataPlan: int
    DataUsage: float
    CustServCalls: int
    DayMins: float
    DayCalls: int
    MonthlyCharge: float
    OverageFee: float
    RoamMins: float


pipeline = PredictPipeline()


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}


@app.post("/predict")
def predict(data: CustomerData):

    features = data.model_dump()

    prediction, probability = pipeline.predict(features)

    result = "Churn" if prediction == 1 else "No Churn"

    return {
        "prediction": result,
        "churn_probability": round(float(probability), 4)
    }