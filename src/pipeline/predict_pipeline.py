import os
import joblib
import pandas as pd


class PredictPipeline:

    def __init__(self):
        self.model_path = os.path.join(
            "artifacts",
            "model.pkl"
        )

        self.preprocessor_path = os.path.join(
            "artifacts",
            "preprocessor.pkl"
        )

        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(
            self.preprocessor_path
        )

    def predict(self, features):

        # Convert input into DataFrame
        input_df = pd.DataFrame([features])

        # Apply the same preprocessing used during training
        input_scaled = self.preprocessor.transform(
            input_df
        )

        # Make prediction
        prediction = self.model.predict(
            input_scaled
        )

        probability = self.model.predict_proba(
            input_scaled
        )[:, 1]

        return prediction[0], probability[0]