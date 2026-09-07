import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.database.mongodb import collection
from src.logger import logging
from src.exception import CustomException


def fetch_data_from_mongodb():
    try:
        logging.info("Fetching data from MongoDB")

        data = list(collection.find({}, {"_id": 0}))
        df = pd.DataFrame(data)

        logging.info("Data successfully fetched from MongoDB")

        return df

    except Exception as e:
        logging.error("Error while fetching data from MongoDB")
        raise CustomException(e, sys)


def initiate_data_ingestion():

    try:
        # Fetch data from MongoDB
        df = fetch_data_from_mongodb()

        logging.info("Splitting data into train and test sets")

        train_set, test_set = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df["Churn"]
        )

        # Create artifacts folder
        os.makedirs("artifacts", exist_ok=True)

        train_path = os.path.join("artifacts", "train.csv")
        test_path = os.path.join("artifacts", "test.csv")

        # Save datasets
        train_set.to_csv(train_path, index=False)
        test_set.to_csv(test_path, index=False)

        logging.info(f"Train data at {train_path}  and Test data at {test_path} saved successfully")

        return train_path, test_path

    except Exception as e:
        logging.error("Error occurred during data ingestion")
        raise CustomException(e, sys)