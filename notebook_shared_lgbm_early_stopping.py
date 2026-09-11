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
        volume_cols = [col for col in self.df_to_use.columns if "volume" in col]
        df.drop(columns=volume_cols, inplace=True)
        print("After removing volume features: ", df.shape)  # len(volume_cols)
        # Remove some features that I consider it unuseful

        unuseful_cols = [
            col
            for col in self.df_to_use.columns.to_list()
            if "agent" in col
            or "senders" in col
            or "recipients" in col
            or "merchants" in col
            or "companies" in col
            or "bank_banks" in col
        ]
        df.drop(columns=unuseful_cols, inplace=True)
        print("After removing unuseful features: ", df.shape)

    def tune_lgbm(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        pos_weight=2.8333333333333335,
        random_state=42,
        early_stopping_rounds=100,
        eval_log_period=50,
    ):
        """
        Tune and train LightGBM with early stopping.

        Parameters
        ----------
        X_train, y_train:
            Training data used for hyperparameter search and final fitting.

        X_val, y_val:
            A validation set used for early stopping and for logging the
            train/validation metrics at every boosting iteration.

            If X_val/y_val are not supplied, a stratified 20% split is created
            from X_train/y_train. For a proper experiment, it is preferable to
            pass your existing held-out validation set explicitly.

        pos_weight:
            LightGBM scale_pos_weight for the positive class.

        early_stopping_rounds:
            Number of consecutive rounds without validation improvement before
            training stops.

        eval_log_period:
            Print evaluation metrics every N boosting rounds.

        Returns
        -------
        LGBMClassifier
            The best LightGBM model found by CV and then refit with early
            stopping. The returned model contains:
                - best_iteration_
                - best_score_
                - evals_result_
                - feature_importances_
        """
        from sklearn.model_selection import train_test_split
        import lightgbm as lgb

        # ---------------------------------------------------------------
        # 1. Create/accept a validation set for early stopping
        # ---------------------------------------------------------------
        if X_val is None or y_val is None:
            X_fit, X_val, y_fit, y_val = train_test_split(
                X_train,
                y_train,
                test_size=0.20,
                stratify=y_train,
                random_state=random_state,
            )
            print("No validation set supplied. Created a stratified 20% validation split from X_train.")
        else:
            X_fit, y_fit = X_train, y_train

        # ---------------------------------------------------------------
        # 2. Hyperparameter search
        #
        # Keep n_estimators high enough for early stopping to determine
        # the effective number of trees later.
        # ---------------------------------------------------------------
        param_grid = {
            "scale_pos_weight": [pos_weight],
            "max_depth": [7],
            "num_leaves": [15],
            "colsample_bytree": [0.5],
            "reg_alpha": [10],
            "reg_lambda": [20],
            "learning_rate": [0.05],
        }

        base_lgbm = LGBMClassifier(
            objective="binary",
            n_estimators=2000,
            random_state=random_state,
            importance_type="gain",
            verbosity=-1,
            device="cpu",
            n_jobs=1,
        )

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=random_state,
        )

        grid = GridSearchCV(
            estimator=base_lgbm,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=4,
            verbose=1,
            refit=False,
        )

        grid.fit(X_fit, y_fit)

        print(f"Best Parameters from CV: {grid.best_params_}")
        print(f"Best CV ROC-AUC: {grid.best_score_:.6f}")

        # ---------------------------------------------------------------
        # 3. Refit the best configuration with early stopping.
        #
        # We deliberately evaluate BOTH train and validation sets so that
        # the overfitting trajectory is visible.
        # ---------------------------------------------------------------
        best_params = grid.best_params_

        lgb_model = LGBMClassifier(
            **best_params,
            objective="binary",
            n_estimators=2000,
            random_state=random_state,
            importance_type="gain",
            verbosity=-1,
            device="cpu",
            n_jobs=1,
        )

        evals_result = {}

        lgb_model.fit(
            X_fit,
            y_fit,
            eval_X=X_val,
            eval_y=y_val,
            # eval_set=[
            #     (X_fit, y_fit),
            #     (X_val, y_val),
            # ],
            eval_names=["train", "valid"],
            eval_metric=["binary_logloss"],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=early_stopping_rounds,
                    first_metric_only=False,
                    verbose=True,
                ),
                lgb.log_evaluation(period=eval_log_period),
                lgb.record_evaluation(evals_result),
            ],
        )

        # Keep an explicit copy on the model as well. This is useful when
        # inspecting the notebook after training.
        # lgb_model.evals_result_ = evals_result

        # ---------------------------------------------------------------
        # 4. Report the final train/validation metrics at the selected
        #    early-stopping iteration.
        # ---------------------------------------------------------------
        train_probs = lgb_model.predict_proba(X_fit)[:, 1]
        val_probs = lgb_model.predict_proba(X_val)[:, 1]

        train_auc = roc_auc_score(y_fit, train_probs)
        val_auc = roc_auc_score(y_val, val_probs)

        train_loss = log_loss(y_fit, train_probs)
        val_loss = log_loss(y_val, val_probs)

        print("\n" + "=" * 70)
        print("LIGHTGBM EARLY-STOPPING SUMMARY")
        print("=" * 70)
        print(f"Best iteration : {lgb_model.best_iteration_}")
        print(f"Train ROC-AUC  : {train_auc:.6f}")
        print(f"Valid ROC-AUC  : {val_auc:.6f}")
        print(f"ROC-AUC gap    : {train_auc - val_auc:.6f}")
        print(f"Train Log Loss : {train_loss:.6f}")
        print(f"Valid Log Loss : {val_loss:.6f}")
        print("=" * 70)

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
        X_train.drop(columns=non_performing_features2, inplace=True)
        if X_val is not None:
            X_val.drop(columns=non_performing_features2, inplace=True)

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
        train_auc = roc_auc_score(y_train, train_m_preds)
        val_auc = roc_auc_score(y_val, val_m_preds)
        train_loss = log_loss(y_train, train_m_preds)
        val_loss = log_loss(y_val, val_m_preds)

        print(f"Train ROC-AUC : {train_auc:.6f}")
        print(f"Val ROC-AUC   : {val_auc:.6f}")
        print(f"ROC-AUC gap   : {train_auc - val_auc:.6f}")
        print(f"Train Log Loss: {train_loss:.6f}")
        print(f"Val Log Loss  : {val_loss:.6f}")

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
