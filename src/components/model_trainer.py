import os
import sys
import json
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(PROJECT_ROOT, "artifacts", "model.pkl")
    model_report_path = os.path.join(PROJECT_ROOT, "artifacts", "model_report.json")


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):

        try:

            logging.info("Training models")

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(class_weight='balanced'),
                "Gradient Boosting": GradientBoostingClassifier(),
                "KNN": KNeighborsClassifier(),
                "SGD": SGDClassifier(loss='log_loss'),
                "AdaBoost": AdaBoostClassifier(),
                "XGBoost": XGBClassifier(eval_metric='mlogloss'),
                "LightGBM": LGBMClassifier(),
                "CatBoost": CatBoostClassifier(verbose=0, allow_writing_files=False)
            }


            params = {

                "Logistic Regression": {
                    "C": [0.1, 1, 10]
                },

                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 20]
                },

                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [100, 200]
                },

                "KNN": {
                    "n_neighbors": [3, 5, 7]
                },

                "SGD": {
                    "alpha": [0.0001, 0.001]
                },

                "AdaBoost": {
                    "n_estimators": [100, 200]
                },

                "XGBoost": {
                    "learning_rate": [0.05, 0.1],
                    "n_estimators": [200, 300]
                },

                "LightGBM": {
                    "learning_rate": [0.05, 0.1],
                    "n_estimators": [200, 300]
                },

                "CatBoost": {
                    "depth": [4, 6],
                    "learning_rate": [0.05, 0.1]
                }
            }


            model_report = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )


            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]


            logging.info(f"Best model found: {best_model_name}")


            # Fit best model again
            best_model.fit(X_train, y_train)


            predictions = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)
            macro_f1 = f1_score(y_test, predictions, average="macro")


            logging.info(f"Accuracy: {accuracy}")
            logging.info(f"Macro F1: {macro_f1}")


            # Generate evaluation report
            report = classification_report(y_test, predictions, output_dict=True)
            cm = confusion_matrix(y_test, predictions)


            evaluation_results = {
                "best_model": best_model_name,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "classification_report": report,
                "confusion_matrix": cm.tolist()
            }


            # Save evaluation report
            with open(self.model_trainer_config.model_report_path, "w") as f:
                json.dump(evaluation_results, f, indent=4)


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )


            return macro_f1


        except Exception as e:
            raise CustomException(e, sys)