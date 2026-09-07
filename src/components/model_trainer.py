import os
import sys
import joblib

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.logger import logging
from src.exception import CustomException


class ModelTrainer:

    def __init__(self):
        self.model_path = os.path.join(
            "artifacts",
            "model.pkl"
        )

    def initiate_model_training(
        self,
        X_train,
        y_train,
        X_test,
        y_test
    ):

        try:
            # Apply SMOTE only to training data
            logging.info("Applying SMOTE")

            smote = SMOTE(random_state=42)

            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

            logging.info("SMOTE applied successfully")

            print("Before SMOTE:")
            print(y_train.value_counts())

            print("\nAfter SMOTE:")
            print(y_train_smote.value_counts())

            # Models
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

            # Train models
            best_model = None
            best_f1 = 0

            for name, model in models.items():

                logging.info(f"Training {name}")

                # Train on SMOTE data
                model.fit(
                    X_train_smote,
                    y_train_smote
                )

                # Test on ORIGINAL test data
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

                # Metrics
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

                # Select best model based on F1
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model

            # Save best model
            joblib.dump(best_model, self.model_path)

            logging.info(f"Best model: {type(best_model).__name__}")
            logging.info(f"Best F1 Score: {round(best_f1, 4)}")

            print("\n----------------------------")
            print("Best Model:", type(best_model).__name__)
            print("Best F1 Score:", round(best_f1, 4))
            print("----------------------------")

            return best_model

        except Exception as e:
            logging.error("Error occurred during model training")
            raise CustomException(e, sys)