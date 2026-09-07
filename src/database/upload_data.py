import pandas as pd
from src.database.mongodb import collection

import sys
from src.logger import logging
from src.exception import CustomException

import os
from dotenv import load_dotenv
load_dotenv()

path = os.getenv("DATA_PATH")

try:
    # Reads the Data
    logging.info("Reading data from CSV")
    df = pd.read_csv(path)

    # Converts the data to Dataframe
    logging.info("Converting data to records")
    data = df.to_dict("records")

    # Inserts the data to MongoDB
    logging.info("Inserting data into MongoDB")
    collection.insert_many(data)

    logging.info("Data inserted successfully")

except Exception as e:
    logging.error("Error while inserting data into MongoDB")
    raise CustomException(e, sys)