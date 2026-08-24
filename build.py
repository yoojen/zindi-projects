import os  # noqa: I001
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
        self.x_diff = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
        self.x_var = 17.5

    def _compute_slopes(self, df: pd.DataFrame, prefix: str) -> np.ndarray:
        """Calculates linear trajectory slopes across M6->M1 in a single vectorized matrix op."""
        cols = [f"m{i}_{prefix}" for i in range(6, 0, -1)]
        if not all(col in df.columns for col in cols):
            return np.zeros(len(df))

        matrix = df[cols].fillna(0).values
        matrix_mean = matrix.mean(axis=1, keepdims=True)
        slopes = np.sum((matrix - matrix_mean) * self.x_diff, axis=1) / self.x_var
        return slopes

    def transform(
        self, df: pd.DataFrame, is_train: bool = True
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """Generates hybrid feature matrix."""
        data = df.copy()

        # Extract Target and drop ID if present
        y = (
            data.pop(self.target_col)
            if (is_train and self.target_col in data)
            else None
        )
        if self.id_col in data.columns:
            data.drop(columns=[self.id_col], inplace=True)

        # 1. Vectorized Trend & Acceleration Slopes
        data["bal_slope_6m"] = self._compute_slopes(data, "daily_avg_bal")
        data["deposit_slope_6m"] = self._compute_slopes(data, "deposit_total_value")
        data["withdraw_slope_6m"] = self._compute_slopes(data, "withdraw_total_value")

        # 2. Key Recency & Liquidity Stress Ratios
        epsilon = 1e-6
        if "m1_daily_avg_bal" in data.columns and "m6_daily_avg_bal" in data.columns:
            data["bal_recency_ratio"] = (data["m1_daily_avg_bal"] + epsilon) / (
                data["m6_daily_avg_bal"] + epsilon
            )

        if "m1_withdraw_total_value" in data.columns:
            data["m1_cashout_intensity"] = data["m1_withdraw_total_value"] / (
                data["m1_daily_avg_bal"] + epsilon
            )

        # 6-Month Aggregates
        bal_cols = [f"m{i}_daily_avg_bal" for i in range(1, 7)]
        if all(c in data.columns for c in bal_cols):
            data["avg_daily_bal_6m"] = data[bal_cols].mean(axis=1)
            data["bal_volatility_6m"] = data[bal_cols].std(axis=1)

        # 3. Log Transformation for Heavy-Tailed Values
        value_cols = [c for c in data.columns if "total_value" in c or "bal" in c]
        for col in value_cols:
            if data[col].dtype in ["float64", "int64"]:
                data[f"{col}_log"] = np.log1p(np.maximum(0, data[col]))

        # 4. One-Hot Encoding for Categorical Demographics
        existing_cats = [c for c in self.categorical_cols if c in data.columns]
        data = pd.get_dummies(data, columns=existing_cats, drop_first=True)

        return data, y

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
        cat_probs = self.cat_model.predict_proba(X_test)[:, 1]
        lr_probs = self.log_reg_model.predict_proba(X_test)[:, 1]

        # Weighted Ensemble Average
        weighed_probs = 0.45 * lgb_probs + 0.45 * cat_probs + 0.10 * lr_probs
        submission = pd.DataFrame({"ID": raw_test_df["ID"], "Target": weighed_probs})
        return weighed_probs, submission

    def split_dataset(self, raw_df, val_size=0.2, random_state=42):
        # 1. Perform Stratified Train/Val Split on raw data
        raw_train, raw_val = train_test_split(
            raw_df,
            test_size=val_size,
            stratify=raw_df[self.target_col],
            random_state=random_state,
        )
        print(
            f"Dataset Split: Train = {len(raw_train)} rows, Val = {len(raw_val)} rows"
        )

        # 2. Transform Train & Validation Feature Matrices
        X_train, y_train = self.feature_pipeline.transform(raw_train, is_train=True)
        X_val, y_val = self.feature_pipeline.transform(raw_val, is_train=True)

        # Align Validation features with Training columns to prevent mismatches
        self.fitted_columns = X_train.columns.tolist()
        X_val = X_val.reindex(columns=self.fitted_columns, fill_value=0)

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
            "max_depth": [3, 5, 7],
            "num_leaves": [15, 31, 63],
            "min_child_samples": [20, 50, 100],
            # 3. Regularization & Subsampling
            "subsample": [0.7, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.9, 1.0],
            "reg_alpha": [0.0, 0.1, 1.0],
            "reg_lambda": [0.0, 1.0, 5.0],
            "learning_rate": [0.02, 0.05],
            "n_estimators": [300],
        }

        lgb = LGBMClassifier(
            random_state=random_state,
            verbosity=-1,
            device=os.environ["device"],
            n_jobs=1,
        )

        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state)

        # Target F1 or Average Precision (PR-AUC) instead of generic accuracy
        grid = GridSearchCV(
            estimator=lgb,
            param_grid=param_grid,
            scoring="f1",  # Or 'average_precision' / 'roc_auc'
            cv=cv,
            n_jobs=4,
            verbose=1,
        )

        grid.fit(X_train, y_train)

        print(f"Best Parameters: {grid.best_params_}")
        print(f"Best CV Score: {grid.best_score_:.4f}")

        self.lgb_model = grid.best_estimator_
        return self.lgb_model

    def run_predictions(self, X_val, threshold: float = 0.5):

        # X_val_scaled = self.scaler.transform(X_val.fillna(0))
        lgb_val_probs = self.lgb_model.predict_proba(X_val)[:, 1]
        # cat_val_probs = self.cat_model.predict_proba(X_val)[:, 1]
        # lr_val_probs = self.log_reg_model.predict_proba(X_val_scaled)[:, 1]

        # + 0.40 * cat_val_probs + 0.10 * lr_val_probs
        self.blended_val_probs = lgb_val_probs
        val_preds = (self.blended_val_probs >= threshold).astype(int)

        return val_preds

    def fit_and_evaluate(
        self,
        csv_path: str,
        val_size: float = 0.2,
        random_state: int = 42,
        threshold: float = 0.5,
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

        self.tune_lgbm(X_train, y_train, random_state)
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
        val_preds = self.run_predictions(X_val, threshold)

        auc = roc_auc_score(y_val, self.blended_val_probs)
        print(f"Validation ROC-AUC Score: {auc:.6f}")
        print("\nClassification Report:")
        print(classification_report(y_val, val_preds))
        print("====================================================")

        with open("training_metrics.pkl", "wb") as f:
            pickle.dump(
                {
                    "X_train": X_train,
                    "y_train": y_train,
                    "X_val": X_val,
                    "y_val": y_val,
                    "val_preds": val_preds,
                },
                f,
            )

        # Save trained model
        self.fitted_columns = X_train.columns.tolist()
        with open("model_pipeline.pkl", "wb") as f:
            pickle.dump(self, f)

        self.lgb_model.booster_.save_model("lgb_model.txt")
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
        return 0.45 * lgb_probs + 0.45

    def predict(self, test_df_or_path, threshold: float = 0.5) -> np.ndarray:
        """Predicts binary labels (0 or 1) based on custom decision threshold."""
        probs = self.predict_proba(test_df_or_path)
        return (probs >= threshold).astype(int)

    # def export_submission_csv(self):
