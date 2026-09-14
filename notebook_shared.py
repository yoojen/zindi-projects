import json
import re

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from build import FeaturePipeline


class SharedNotebook:
    def __init__(self, algorithm: LGBMClassifier, fp=FeaturePipeline, threshold: float = 0.5, is_train: bool = True):
        self.feature_pipeline = fp()
        self.algorithm = algorithm
        self.threshold = threshold

        self.train_df = None
        self.test_df = None
        self.is_train = is_train

    def read_dataset(self, csv_path: str, is_train: bool = True):
        if is_train:
            self.train_df = pd.read_csv(csv_path)
        else:
            print("Reaing test dataset")
            self.test_df = pd.read_csv(csv_path)

    @property
    def df_to_use(self):
        print("Is self.test available: ", self.test_df is not None)
        print("Is it training mode: ", self.is_train)
        if self.is_train and self.train_df is not None:
            return self.train_df
        elif not self.is_train and self.test_df is not None:
            return self.test_df
        else:
            raise ValueError("No dataframe to use")

    def calculate_monthovermonth(self, keywords: list):
        running_df = pd.DataFrame()
        all_affected_cols = []

        for keyword in keywords:
            keyword_df, affected_cols = self.feature_pipeline.month_over_month_calculation(keyword, self.df_to_use)

            all_affected_cols += affected_cols
            # If running df is empty replace what it hold
            try:
                if running_df.empty:
                    running_df = keyword_df
                else:
                    running_df = pd.concat([running_df, keyword_df], axis=1)
            except Exception as e:
                print("Exception:\n=====================\n:", str(e))

        return running_df

    def find_and_remove_highest_amt_features(self):
        pattern = r".+highest_amount$"
        df_cols = self.df_to_use.columns.to_list()
        affected_cols = []
        for s in df_cols:
            match = re.search(pattern, s)
            if match:
                # Extract the specific keyword that triggered the match
                keyword_found = match.group(0)
                affected_cols.append(keyword_found)
            else:
                pass

        self.df_to_use.drop(columns=affected_cols, inplace=True)
        return affected_cols

    def remove_correlated_features(self, df: pd.DataFrame):
        """Removes features with a correlation higher than 0.80.
        Never return anything, instead drops the features in-memory"""
        # Create a correlation matrix for your engineered features
        corr_matrix = df.corr().abs()

        # Select the upper triangle of the matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with a correlation higher than 0.80
        to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]

        # Safely drop only the strictly identical redundancies
        df.drop(columns=to_drop, inplace=True)

    def define_features_to_remove(self, df: pd.DataFrame):
        """Removes features that I consider it unuseful. It use self.train_df to get the columns,
        but remove the defined columns from the submitted dataframe"""
        # Remove volume feature
        # volume_cols = [col for col in self.df_to_use.columns if "volume" in col]
        # df.drop(columns=volume_cols, inplace=True)
        # print("After removing volume features: ", df.shape, "<<>>", len(volume_cols))  # len(volume_cols)
        # Remove some features that I consider it unuseful

        unuseful_cols = [
            col
            for col in self.df_to_use.columns.to_list()
            # if "agents" in col
            if "senders" in col
            or "recipients" in col
            # or "merchants" in col
            # or "companies" in col
            or "bank_banks" in col
        ]
        df.drop(columns=unuseful_cols, inplace=True)
        print("After removing unuseful features: ", df.shape)

    def tune_lgbm(self, X_train, y_train, pos_weight=2.8333333333333335, random_state=42):
        """Tunes LightGBM hyperparameters using Stratified K-Fold to balance Precision and Recall."""
        print(X_train.shape, y_train.shape)
        # Currently best performer - Sept 14
        param_grid = {
            # 1. Direct control over positive class weight (scale down to boost precision)
            "scale_pos_weight": [
                # pos_weight * 0.5,
                # pos_weight * 0.75,
                pos_weight,
            ],
            # 2. Tree structural constraints (controls overfitting/false positives)
            "max_depth": [7],
            # "max_depth": [6],
            "num_leaves": [15],
            # "min_data_in_leaf": [1000],
            # 3. Regularization & Subsampling
            "colsample_bytree": [0.5],
            "reg_alpha": [10],
            "reg_lambda": [20],
            "learning_rate": [0.01],
            "objective": ["binary"],
            # "n_estimators": [400],
            "n_estimators": [3000],
        }

        lgb = LGBMClassifier(
            random_state=random_state,
            importance_type="gain",
            verbosity=-1,
            device="cpu",
            n_jobs=1,
            metric="binary_logloss",
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        # Target F1 or Average Precision (PR-AUC) instead of generic accuracy
        grid = GridSearchCV(
            estimator=lgb,
            param_grid=param_grid,
            scoring="roc_auc",  # Or 'average_precision' / 'roc_auc'
            cv=cv,
            n_jobs=4,
            verbose=1,
        )

        grid.fit(X_train, y_train)

        print(f"Best Parameters: {grid.best_params_}")
        print(f"Best CV Score: {grid.best_score_:.4f}")

        lgb_model = grid.best_estimator_
        return lgb_model

    def tune_random_forest(self, X_train, y_train, pos_weigth=2.8333333333333335, random_state=42):
        # param_grid = {
        #     "n_estimators": [500, 1000],
        #     "max_depth": [5, 7, 9],
        #     "max_features": ["sqrt", "log2"],
        #     "criterion": ["gini", "log_loss"],
        #     "max_leaf_nodes": [300, 500, 700],
        #     "bootstrap": [True],
        #     "class_weight": ["balanced"],
        #     "oob_score": [True, False],
        # }
        param_grid = {
            "bootstrap": [True],
            "class_weight": ["balanced"],
            "criterion": ["gini"],
            "max_depth": [9],
            "max_features": ["sqrt"],
            "max_leaf_nodes": [100, 300],
            "n_estimators": [1000],
            "oob_score": [True],
        }

        rf = RandomForestClassifier(random_state=random_state, n_jobs=1)

        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state)

        # Target F1 or Average Precision (PR-AUC) instead of generic accuracy
        grid = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            scoring="neg_log_loss",  # Or 'average_precision' / 'roc_auc'
            cv=cv,
            n_jobs=4,
            verbose=1,
        )

        grid.fit(X_train, y_train)

        print(f"Best Parameters: {grid.best_params_}")
        print(f"Best CV Score: {grid.best_score_:.4f}")

        rf_model = grid.best_estimator_
        return rf_model

    @staticmethod
    def find_and_remove_mom_features(df: pd.DataFrame):
        """Remove all month over month features (which were manually calculated), they seem to add nothing to the model"""
        import re

        pattern = r".+_mom$"
        mom_features = [col for col in df.columns if re.search(pattern, col)]
        df.drop(columns=mom_features, inplace=True)

    @staticmethod
    def remove_features_definedby_algorithm_gain(X_train: pd.DataFrame, X_val: pd.DataFrame | None):
        """Load and remove previously features which were not useful based on feature importance gains"""
        with open("non_performing_features.json", "r") as f:
            non_performing_features = json.load(f)

        with open("A_non_performing_features2.json", "r") as f:
            non_performing_features2 = json.load(f)

        X_train.drop(columns=non_performing_features, inplace=True)
        if X_val is not None:
            X_val.drop(columns=non_performing_features, inplace=True)

        # Remvoe the version2 which were based on the cummulative and relative gain
        # X_train.drop(columns=non_performing_features2, inplace=True)
        # if X_val is not None:
        #     X_val.drop(columns=non_performing_features2, inplace=True)

    def remove_netflow_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame | None):
        # Remove some features and see how model performs (Removing these columns improved the ROC and LOG LOSS)
        x_cols = [
            "m1_net_flow",
            "m2_net_flow",
            "m3_net_flow",
            "m4_net_flow",
            "m5_net_flow",
            "m6_net_flow",
        ]
        X_train.drop(columns=x_cols, inplace=True)
        if X_val is not None:
            X_val.drop(columns=x_cols, inplace=True)

    @staticmethod
    def run_predictions(model, X_train, X_val, y_train, y_val):
        train_m_preds = model.predict_proba(X_train)[:, 1]  # type: ignore
        val_m_preds = model.predict_proba(X_val)[:, 1]  # type: ignore
        print("Train ROC: ", roc_auc_score(y_train, train_m_preds))
        print("Val ROC: ", roc_auc_score(y_val, val_m_preds))
        print("Val Log Loss: ", log_loss(y_val, val_m_preds))

        return train_m_preds, val_m_preds

    @staticmethod
    def build_cm(y_val, val_m_preds):
        cm = confusion_matrix(y_val, val_m_preds > 0.5)
        # First show the classification report
        print("Classification report: \n", classification_report(y_val, val_m_preds > 0.5), end="\n=======\n")
        return ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1]).plot(cmap="Blues")

    @staticmethod
    def calibrate_model(model, X_train, y_train):
        calibrated_lgb = CalibratedClassifierCV(estimator=model, method="isotonic", cv=5)
        calibrated_lgb.fit(X_train, y_train)
        return calibrated_lgb

    @staticmethod
    def run_calibrated_predictions(calibrated_lgb, X_val):
        val_pred = calibrated_lgb.predict_proba(X_val)[:, 1]
        return val_pred

    @staticmethod
    def display_val_prediction_stats(val_pred, y_val):
        print("Val ROC: ", roc_auc_score(y_val, val_pred))
        print("Val Log Loss: ", log_loss(y_val, val_pred))

    @staticmethod
    def save_submission_file(test_df, test_probs):
        submission = pd.DataFrame({"ID": test_df["ID"], "Target": test_probs})
        submission.to_csv("submission.csv", index=False)
        print("Submission file saved!")

        return submission

    @staticmethod
    def save_model(model, model_name: str):
        filename = f"{model_name}.pkl"
        import pickle

        with open(filename, "wb") as f:
            pickle.dump(model, f)
