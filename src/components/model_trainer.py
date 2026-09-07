import os
import sys
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.logger import logging
from src.exception import CustomException


class ModelTrainer:

    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")

    def initiate_model_training(
        self,
        X_train,
        y_train,
        X_test,
        y_test
    ):

        try:
            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000
                ),

                "Random Forest": RandomForestClassifier(
                    n_estimators=200,
                    random_state=42
                ),

                "Gradient Boosting": GradientBoostingClassifier(
                    random_state=42
                ),

                "XGBoost": XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42,
                    eval_metric="logloss"
                ),

                "LightGBM": LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42,
                    verbosity=-1
                ),

                "CatBoost": CatBoostClassifier(
                    iterations=200,
                    learning_rate=0.05,
                    depth=5,
                    random_seed=42,
                    verbose=False
                )
            }

            best_model = None
            best_f1 = 0

            for name, model in models.items():

                logging.info(f"Training {name}")

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_prob)

                print(f"\n{name}")
                print(f"Accuracy : {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall   : {recall:.4f}")
                print(f"F1 Score : {f1:.4f}")
                print(f"ROC-AUC  : {roc_auc:.4f}")

                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model

            joblib.dump(best_model, self.model_path)

            logging.info("Best model saved successfully")

            print("\nBest model:", type(best_model).__name__)
            print("Best F1 Score:", round(best_f1, 4))

            return best_model

        except Exception as e:
            logging.error("Error occurred during model training")
            raise CustomException(e, sys)

if __name__ == "__main__":

    from src.components.data_transformation import DataTransformation

    train_path = os.path.join("artifacts", "train.csv")
    test_path = os.path.join("artifacts", "test.csv")

    transformation = DataTransformation()

    X_train, y_train, X_test, y_test, preprocessor = \
        transformation.initiate_data_transformation(
            train_path,
            test_path
        )

    trainer = ModelTrainer()

    trainer.initiate_model_training(
        X_train,
        y_train,
        X_test,
        y_test
    )