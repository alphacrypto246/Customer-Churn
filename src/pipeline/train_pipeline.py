import os

from src.components.data_ingestions import initiate_data_ingestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:

    def run_pipeline(self):

        # 1. Data Ingestion
        train_path, test_path = initiate_data_ingestion()

        # 2. Data Transformation
        data_transformation = DataTransformation()

        X_train, y_train, X_test, y_test, preprocessor = (
            data_transformation.initiate_data_transformation(train_path, test_path
            )
        )

        # 3. Model Training
        model_trainer = ModelTrainer()

        model_trainer.initiate_model_training(
            X_train,
            y_train,
            X_test,
            y_test
        )


if __name__ == "__main__":

    pipeline = TrainPipeline()

    pipeline.run_pipeline()