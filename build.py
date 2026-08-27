import os  # noqa: I001
import re
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from xgboost import XGBClassifier

load_dotenv()

# ==========================================
# 1. FEATURE PIPELINE CLASS
# ==========================================

if "device" not in os.environ:
    os.environ["device"] = "cpu"


class FeaturePipeline:
    """Transforms raw monthly transaction data into a robust hybrid feature set.

    Handles time-series slope/trend extraction, liquidity stress metrics, and log
    transforms.
    """

    def __init__(self, target_col="liquidity_stress_next_30d", id_col="ID"):
        self.target_col = target_col
        self.id_col = id_col
        self.categorical_cols = [
            "gender",
            "region",
            "smartphone",
            "segment",
            "earning_pattern",
        ]

        # Constants for vector slope calculation across M6->M1
        # self.x_diff_6months = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
        # self.x_diff_3months = np.array([-1.5, -0.5, 0.5])
        # self.x_var = 17.5

        self.x_diff_6m = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
        self.x_var_6m = 17.5
        self.x_diff_3m = np.array([-1.0, 0.0, 1.0])
        self.x_var_3m = 2.0

    def _compute_slopes(
        self, df: pd.DataFrame, prefix: str, months: int = 6
    ) -> np.ndarray:
        """Calculates linear slopes across M6->M1 or M3->M1."""
        cols = [f"m{i}_{prefix}" for i in range(months, 0, -1)]
        if not all(c in df.columns for c in cols):
            return np.zeros(len(df))

        matrix = df[cols].fillna(0).values
        matrix_mean = matrix.mean(axis=1, keepdims=True)

        x_diff = self.x_diff_6m if months == 6 else self.x_diff_3m
        x_var = self.x_var_6m if months == 6 else self.x_var_3m

        return np.sum((matrix - matrix_mean) * x_diff, axis=1) / x_var

    def transform(
        self, df: pd.DataFrame, is_train: bool = True
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        data = df.copy()
        epsilon = 1e-6

        # 1. Pop Target and drop ID
        y = (
            data.pop(self.target_col)
            if (is_train and self.target_col in data)
            else None
        )
        if self.id_col in data.columns:
            data.drop(columns=[self.id_col], inplace=True)

        # 2. Compute Consolidated Inflows and Outflows per Month (M1 to M6)
        inflow_channels = ["deposit", "received", "transfer_from_bank"]
        outflow_channels = ["withdraw", "paybill", "merchantpay", "mm_send"]

        # for m in range(1, 7):
        #     # Sum monthly total values
        #     inf_cols = [
        #         f"m{m}_{ch}_total_value"
        #         for ch in inflow_channels
        #         if f"m{m}_{ch}_total_value" in data.columns
        #     ]
        #     out_cols = [
        #         f"m{m}_{ch}_total_value"
        #         for ch in outflow_channels
        #         if f"m{m}_{ch}_total_value" in data.columns
        #     ]

        #     data[f"m{m}_total_inflow"] = data[inf_cols].sum(axis=1) if inf_cols else 0.0
        #     data[f"m{m}_total_outflow"] = (
        #         data[out_cols].sum(axis=1) if out_cols else 0.0
        #     )
        #     data[f"m{m}_net_flow"] = (
        #         data[f"m{m}_total_inflow"] - data[f"m{m}_total_outflow"]
        #     )

        new_features = {}
        for m in range(1, 7):
            # Sum monthly total values
            inf_cols = [
                f"m{m}_{ch}_total_value"
                for ch in inflow_channels
                if f"m{m}_{ch}_total_value" in data.columns
            ]
            out_cols = [
                f"m{m}_{ch}_total_value"
                for ch in outflow_channels
                if f"m{m}_{ch}_total_value" in data.columns
            ]

            # 2. Assign calculations to the dictionary instead of 'data'
            total_inflow = data[inf_cols].sum(axis=1) if inf_cols else 0.0
            total_outflow = data[out_cols].sum(axis=1) if out_cols else 0.0

            new_features[f"m{m}_total_inflow"] = total_inflow
            new_features[f"m{m}_total_outflow"] = total_outflow
            new_features[f"m{m}_net_flow"] = total_inflow - total_outflow

        # 3. Convert all new columns at once and join them horizontally
        new_df = pd.DataFrame(new_features, index=data.index)
        data = pd.concat([data, new_df], axis=1)

        # 3. 3-Month Window Median Aggregations
        m1_3_inflow = [f"m{i}_total_inflow" for i in range(1, 4)]
        m4_6_inflow = [f"m{i}_total_inflow" for i in range(4, 7)]
        m1_3_outflow = [f"m{i}_total_outflow" for i in range(1, 4)]
        m4_6_outflow = [f"m{i}_total_outflow" for i in range(4, 7)]

        data["inflow_median_recent_3m"] = data[m1_3_inflow].median(axis=1)
        data["inflow_median_baseline_3m"] = data[m4_6_inflow].median(axis=1)
        data["outflow_median_recent_3m"] = data[m1_3_outflow].median(axis=1)
        data["outflow_median_baseline_3m"] = data[m4_6_outflow].median(axis=1)

        # 3M Flow Dynamics / Ratios
        data["inflow_drift_3m"] = (data["inflow_median_recent_3m"] + epsilon) / (
            data["inflow_median_baseline_3m"] + epsilon
        )
        data["outflow_drift_3m"] = (data["outflow_median_recent_3m"] + epsilon) / (
            data["outflow_median_baseline_3m"] + epsilon
        )
        data["net_coverage_recent_3m"] = (data["inflow_median_recent_3m"] + epsilon) / (
            data["outflow_median_recent_3m"] + epsilon
        )

        # 4. Slopes (3m and 6m Trajectories)
        data["bal_slope_6m"] = self._compute_slopes(data, "daily_avg_bal", months=6)
        data["bal_slope_3m"] = self._compute_slopes(data, "daily_avg_bal", months=3)
        data["inflow_slope_6m"] = self._compute_slopes(data, "total_inflow", months=6)
        data["outflow_slope_6m"] = self._compute_slopes(data, "total_outflow", months=6)

        # 5. Granular End-Points (M1 & M6) and Ratios
        if "m1_daily_avg_bal" in data.columns and "m6_daily_avg_bal" in data.columns:
            data["bal_m1_m6_ratio"] = (data["m1_daily_avg_bal"] + epsilon) / (
                data["m6_daily_avg_bal"] + epsilon
            )

        if "m1_total_outflow" in data.columns and "m1_daily_avg_bal" in data.columns:
            data["m1_cashout_intensity"] = data["m1_total_outflow"] / (
                data["m1_daily_avg_bal"] + epsilon
            )

        # 6. Log Transformation on Continuous Monetary Features
        monetary_cols = [
            c
            for c in data.columns
            if "total" in c
            or "bal" in c
            or "inflow" in c
            or "outflow" in c
            or "net" in c
        ]
        normalized_monetary_cols = {}
        for col in monetary_cols:
            if data[col].dtype in ["float64", "int64"]:
                # data[f"{col}_log"] = np.log1p(np.maximum(0, data[col]))
                normalized_monetary_cols[f"{col}_log"] = np.log1p(
                    np.maximum(0, data[col])
                )

        # Convert the dictionary to a DataFrame all at once
        new_df = pd.DataFrame(normalized_monetary_cols, index=data.index)

        # Concatenate it horizontally with your original data
        data = pd.concat([data, new_df], axis=1)

        # Remove transformed columns
        # data.drop(columns=monetary_cols, inplace=True, errors="ignore")

        # 7. Drop Redundant Intermediate Monthly Columns (M2, M3, M4, M5)
        # Retain M1 (immediate boundary) and M6 (baseline boundary)
        # intermediate_cols = []
        # for m in [2, 3, 4, 5]:
        #     intermediate_cols.extend(
        #         [c for c in data.columns if c.startswith(f"m{m}_")]
        #     )

        # intermediate_cols = intermediate_cols + monetary_cols
        # data.drop(columns=list(set(intermediate_cols)), inplace=True, errors="ignore")

        # 8. One-Hot Encoding
        existing_cats = [c for c in self.categorical_cols if c in data.columns]
        data = pd.get_dummies(data, columns=existing_cats, drop_first=True)

        return data, y

    def get_bills_cols(self, df):
        # List of target keywords you want to look for in the middle
        keywords = [r".+total_value$", r".+highest_amount$"]

        # Combine them into: .+(?:_paybill_|_invoice_|_receipt_).+
        # The parenthesis () capture the specific keyword that matched
        pattern = rf"(?P<matched_keyword>{'|'.join(keywords)})"

        # Dataframe columns
        df_cols = df.columns.to_list()
        matched_cols = []
        for s in df_cols:
            match = re.search(pattern, s)
            if match:
                # Extract the specific keyword that triggered the match
                keyword_found = match.group("matched_keyword")
                # print(f"Match found! String: '{s}' | Keyword: '{keyword_found}'")
                matched_cols.append(keyword_found)
            else:
                pass

        return matched_cols

    def transform_for_test_set(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        return self.transform(df, is_train=False)


# ==========================================
# 2. MODEL PIPELINE CONSUMER CLASS
# ==========================================


class ModelPipeline:
    """Consumes a single CSV, splits into Train/Val, trains ensemble, and generates predictions."""

    def __init__(self, target_col="liquidity_stress_next_30d"):
        self.target_col = target_col
        self.feature_pipeline = FeaturePipeline(target_col=target_col)
        self.scaler = StandardScaler()
        self.fitted_columns = None

        # Ensemble Models
        self.lgb_model = None
        self.cat_model = None
        self.log_reg_model = None
        self.xgb_model = None

    def test_predict(self, csv_path: str):
        """Loads dataset, performs feature transformation, and generates predictions."""
        print(f"Reading dataset from: {csv_path}")
        raw_test_df = pd.read_csv(csv_path)

        # 1. Transform Test Feature Matrix
        X_test, _ = self.feature_pipeline.transform_for_test_set(raw_test_df)
        print(f"Test Feature Matrix Shape: {X_test.shape}")
        # Reindex columns to match fitted training columns
        if not self.fitted_columns:
            self.fitted_columns = X_test.columns.tolist()
        X_test = X_test.reindex(columns=self.fitted_columns, fill_value=0)

        # Predict from ensemble models
        lgb_probs = self.lgb_model.predict_proba(X_test)[:, 1]
        # cat_probs = self.cat_model.predict_proba(X_test)[:, 1]
        # lr_probs = self.log_reg_model.predict_proba(X_test)[:, 1]

        # Weighted Ensemble Average
        #  + 0.45 * cat_probs + 0.10 * lr_probs
        weighed_probs = lgb_probs
        submission = pd.DataFrame({"ID": raw_test_df["ID"], "Target": weighed_probs})
        return weighed_probs, submission

    def split_dataset(
        self, raw_df, cross_validation=False, val_size=0.2, random_state=42
    ):
        # 1. Perform Stratified Train/Val Split on raw data
        raw_train, raw_val = train_test_split(
            raw_df,
            test_size=val_size,
            stratify=raw_df[self.target_col],
            random_state=random_state,
        )
        print(
            f"Dataset Split: Train = {len(raw_train)} rows, Val = {len(raw_val) if raw_val is not None else None} rows"
        )

        # 2. Transform Train & Validation Feature Matrices
        X_train, y_train = self.feature_pipeline.transform(raw_train, is_train=True)
        # if raw_val is not None:
        X_val, y_val = self.feature_pipeline.transform(raw_val, is_train=True)
        X_val = X_val.reindex(columns=self.fitted_columns, fill_value=0)
        # else:
        #     X_val, y_val = None, None

        # Align Validation features with Training columns to prevent mismatches
        self.fitted_columns = X_train.columns.tolist()

        # Calculate class imbalance ratio for weighted loss
        pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

        return X_train, y_train, X_val, y_val, pos_weight

    def fit_lgb(self, X_train, y_train, pos_weight, random_state=42):
        print("\n--- Training LightGBM Classifier ---")
        self.lgb_model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            scale_pos_weight=pos_weight,
            random_state=random_state,
            verbosity=-1,
        )
        self.lgb_model.fit(X_train, y_train)

    def fit_cat_model(self, X_train, y_train, pos_weight, random_state=42):
        self.cat_model = CatBoostClassifier(
            iterations=300,
            learning_rate=0.03,
            depth=5,
            scale_pos_weight=pos_weight,
            random_seed=random_state,
            verbose=0,
        )
        self.cat_model.fit(X_train, y_train)

    def fit_lgr_model(self, X_train, y_train, random_state):
        X_train_scaled = self.scaler.fit_transform(X_train.fillna(0))
        self.log_reg_model = LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=1000, random_state=random_state
        )
        self.log_reg_model.fit(X_train_scaled, y_train)

    def tune_lgbm(self, X_train, y_train, random_state=42):
        """Tunes LightGBM hyperparameters using Stratified K-Fold to balance Precision and Recall."""
        pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

        param_grid = {
            # 1. Direct control over positive class weight (scale down to boost precision)
            "scale_pos_weight": [
                pos_weight * 0.5,
                pos_weight * 0.75,
                pos_weight,
            ],
            # 2. Tree structural constraints (controls overfitting/false positives)
            "max_depth": [5, 7, 9],
            "num_leaves": [15, 31, 63],
            # "min_data_in_leaf": [50, 100],
            # "min_child_samples": [20, 50, 100],
            # 3. Regularization & Subsampling
            # "subsample": [0.7, 0.9, 1.0],
            # "colsample_bytree": [0.7, 0.9],
            # "reg_alpha": [0.0, 0.1, 1.0],
            # "reg_lambda": [0.0, 1.0, 5.0],
            "learning_rate": [0.02, 0.05],
            "n_estimators": [350, 400],
        }

        lgb = LGBMClassifier(
            random_state=random_state,
            verbosity=-1,
            device=os.environ["device"],
            n_jobs=1,
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

        self.lgb_model = grid.best_estimator_
        return self.lgb_model

    def tune_xgbost(self, X_train, y_train, X_val, y_val, random_state=42):
        pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
        param_grid = {
            # "scale_pos_weight": [pos_weight * 0.5, pos_weight * 0.75, pos_weight],
            "scale_pos_weight": [pos_weight * 0.5],
            "max_depth": [7],
            "learning_rate": [0.02, 0.03, 0.05],
            "objective": ["binary:logistic"],
            "feature_fraction": [0.5, 0.7],
        }

        estimator = XGBClassifier(
            n_jobs=1,
            random_state=random_state,
            device=os.environ["device"],
            eval_metric="logloss",
            early_stopping_rounds=10,
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        # Target F1 or Average Precision (PR-AUC) instead of generic accuracy
        grid = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="roc_auc",  # Or 'average_precision' / 'roc_auc' / 'log_loss'
            cv=cv,
            n_jobs=4,
            verbose=1,
        )

        grid.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        print(f"Best Parameters: {grid.best_params_}")
        print(f"Best CV Score: {grid.best_score_:.4f}")

        self.xgb_model = grid.best_estimator_
        return self.xgb_model

    def run_predictions(self, X_val, is_train=False, threshold: float = 0.5):

        # X_val_scaled = self.scaler.transform(X_val.fillna(0))
        lgb_val_probs = self.xgb_model.predict_proba(X_val)[:, 1]
        # cat_val_probs = self.cat_model.predict_proba(X_val)[:, 1]
        # lr_val_probs = self.log_reg_model.predict_proba(X_val_scaled)[:, 1]

        # + 0.40 * cat_val_probs + 0.10 * lr_val_probs
        self.blended_val_probs = lgb_val_probs
        val_preds = (self.blended_val_probs >= threshold).astype(int)

        # Save probabilities in pkl file for further threshold analysis
        filename = "val_probs.pkl" if not is_train else "train_probs.pkl"
        with open(filename, "wb") as f:
            pickle.dump(self.blended_val_probs, f)
        return val_preds, self.blended_val_probs

    def fit_and_evaluate(
        self,
        csv_path: str,
        val_size: float = 0.2,
        random_state: int = 42,
        threshold: float = 0.5,
        version: int = 1,
    ):
        """Loads dataset, performs stratified train_test_split, trains ensemble models, and prints validation metrics."""
        print(f"Reading dataset from: {csv_path}")
        raw_df = pd.read_csv(csv_path)
        X_train, y_train, X_val, y_val, pos_weight = self.split_dataset(
            raw_df, val_size, random_state
        )

        print(
            f"Features generated: {X_train.shape[1]} columns. Imbalance Ratio:"
            f" {pos_weight:.2f}"
        )

        # 3. Train LightGBM
        # self.fit_lgb(X_train, y_train, pos_weight, random_state)

        # self.tune_lgbm(X_train, y_train, random_state)
        self.tune_xgbost(X_train, y_train, X_val, y_val, random_state)
        # 4. Train CatBoost
        print("--- Training CatBoost Classifier ---")
        # self.fit_cat_model(X_train, y_train, pos_weight, random_state)

        # 5. Train Regularized Logistic Regression
        print("--- Training Regularized Logistic Regression ---")
        # self.fit_lgr_model(X_train, y_train, random_state)
        print("\n[SUCCESS] All Models Fitted!")

        # 6. Evaluate on Validation Set
        print("\n================ VALIDATION RESULTS ================")

        # Blend probabilities
        val_preds, raw_probs = self.run_predictions(X_val, threshold=threshold)

        auc = roc_auc_score(y_val, raw_probs)
        print(f"Validation ROC-AUC Score: {auc:.6f}")
        print("\nClassification Report:")
        print(classification_report(y_val, val_preds))
        print("====================================================\n\n")
        train_pred, raw_train_pred = self.run_predictions(
            X_train, is_train=True, threshold=threshold
        )
        training_auc = roc_auc_score(y_train, raw_train_pred)
        print(f"Training ROC-AUC Score: {training_auc:.6f}")
        print("\nClassification Report:")
        print(classification_report(y_train, train_pred))
        print("====================================================\n\n")

        # 7. Save model pipeline
        filename = f"training_metrics_v{version}_f{X_train.shape[1]}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(
                {
                    "X_train": X_train,
                    "y_train": y_train,
                    # "X_val": X_val,
                    # "y_val": y_val,
                    # "val_preds": val_preds,
                    # "raw_probs": raw_probs,
                },
                f,
            )

        # Save trained model
        self.fitted_columns = X_train.columns.tolist()
        # Find the saved model pipeline and increment the version
        version = 1
        if os.path.exists("model_pipeline_v1.txt"):
            version += 1

        num_features = len(self.fitted_columns)
        # new_model_filename = f"model_pipeline_v{version}_f{num_features}.txt"
        new_model_filename = f"model_pipeline_v{version}_f{num_features}.json"
        # self.lgb_model.booster_.save_model(new_model_filename)
        self.xgb_model.save_model(new_model_filename)
        return self

    def predict_proba(self, test_df_or_path) -> np.ndarray:
        """Generates blended ensemble probability scores for new test data."""
        if isinstance(test_df_or_path, str):
            df = pd.read_csv(test_df_or_path)
        else:
            df = test_df_or_path.copy()

        X, _ = self.feature_pipeline.transform(df, is_train=False)

        # Reindex columns to match fitted training columns
        X = X.reindex(columns=self.fitted_columns, fill_value=0)

        # Predict from ensemble models
        lgb_probs = self.lgb_model.predict_proba(X)[:, 1]
        # cat_probs = self.cat_model.predict_proba(X)[:, 1]

        # X_scaled = self.scaler.transform(X.fillna(0))
        # lr_probs = self.log_reg_model.predict_proba(X_scaled)[:, 1]

        # Weighted Ensemble Average
        # * cat_probs + 0.10 * lr_probs
        return lgb_probs

    def predict(self, test_df_or_path, threshold: float = 0.5) -> np.ndarray:
        """Predicts binary labels (0 or 1) based on custom decision threshold."""
        probs = self.predict_proba(test_df_or_path)
        return (probs >= threshold).astype(int)

    # def export_submission_csv(self):
