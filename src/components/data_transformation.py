import os
import sys
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from src.logger import logging
from src.exception import CustomException

class DataTransformation:

    def __init__(self):
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def initiate_data_transformation(self, train_path, test_path):

        try:
            logging.info("Reading train and test data")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Separate input features and target
            target_column = "Churn"

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            # All columns in this dataset are numerical
            numerical_columns = X_train.columns.tolist()

            # Preprocessor
            preprocessor = ColumnTransformer(
                [
                    (
                        "num_pipeline",
                        StandardScaler(),
                        numerical_columns
                    )
                ]
            )

            logging.info("Applying StandardScaler")

            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            logging.info("Data transformation completed")

            return (
                X_train_scaled,
                y_train,
                X_test_scaled,
                y_test,
                preprocessor
            )

        except Exception as e:
            logging.error("Error occurred during data transformation")
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataTransformation()

    train_path = os.path.join("artifacts", "train.csv")
    test_path = os.path.join("artifacts", "test.csv")

    result = obj.initiate_data_transformation(
        train_path,
        test_path
    )

    print("Data transformation completed")