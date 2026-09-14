import os  # noqa: I001
import re
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, log_loss
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
    @property
    def x_diff_6m(self):
        return np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])

    @property
    def x_var_6m(self):
        return 17.5

    @property
    def x_diff_3m(self):
        return np.array([-1.0, 0.0, 1.0])

    @property
    def x_var_3m(self):
        return 2.0

    def _compute_slopes(self, df: pd.DataFrame, prefix: str, months: int = 6) -> np.ndarray:
        """Calculates linear slopes across M6->M1 or M3->M1."""
        cols = [f"m{i}_{prefix}" for i in range(months, 0, -1)]
        if not all(c in df.columns for c in cols):
            return np.zeros(len(df))

        matrix = df[cols].fillna(0).values
        matrix_mean = matrix.mean(axis=1, keepdims=True)

        x_diff = self.x_diff_6m if months == 6 else self.x_diff_3m
        x_var = self.x_var_6m if months == 6 else self.x_var_3m

        return np.sum((matrix - matrix_mean) * x_diff, axis=1) / x_var

    def get_bal_avg_cols(self, df: pd.DataFrame):
        months = ["m1", "m2", "m3", "m4", "m5", "m6"]

        return [f"{m}_daily_avg_bal" for m in months]

    def transform_merchant_features(self, df: pd.DataFrame):
        recent3_merchants = [f"m{i}_merchantpay_total_value" for i in range(1, 4)]
        old3_merchants = [f"m{i}_merchantpay_total_value" for i in range(4, 7)]
        all_merchants = recent3_merchants + old3_merchants
        all_merchant_high_amount = [f"m{i}_merchantpay_highest_amount" for i in range(1, 7)]

        running_df = pd.DataFrame()
        running_df["merchant_hghamt_6m"] = df[all_merchant_high_amount].mean(axis=1)

        # Calculate coefficient of variance
        running_df["merchant_cv_3mrecent"] = df[recent3_merchants].std(axis=1) / df[recent3_merchants].mean(axis=1)
        running_df["merchant_cv_3mold"] = df[old3_merchants].std(axis=1) / df[old3_merchants].mean(axis=1)
        running_df["merchant_cv_6m"] = df[all_merchants].std(axis=1) / df[all_merchants].mean(axis=1)

        # merchant Volume calculations
        # Mean of recent 3 months and Mean of all 6 months
        running_df["merchant_3mrecent"] = df[recent3_merchants].mean(axis=1)
        running_df["merchant_total_6m"] = df[all_merchants].mean(axis=1)

        # Company merchant Volume calculations vs total merchant volume and total amount spent on bill (total value)
        all_volume = [f"m{i}_merchantpay_volume" for i in range(1, 7)]

        # running_df["merchant_vol_3mrecent"] = df[recent3_volume].mean(axis=1)
        running_df["merchant_vol_6m"] = df[all_volume].mean(axis=1)

        # Merchantpay (unique merchants paid)
        unique_marchents = [f"m{i}_merchantpay_merchants" for i in range(1, 7)]
        running_df["merchant_unique_6m"] = df[unique_marchents].mean(axis=1)

        # Drop all used raw features (without removing them model was slightly better than others)
        df.drop(columns=all_merchants, inplace=True)
        df.drop(columns=all_volume, inplace=True)
        df.drop(columns=all_merchant_high_amount, inplace=True)
        df.drop(columns=unique_marchents, inplace=True)
        # df.drop(columns=unique_marchent, inplace=True)

        # Join two dfs
        return pd.concat([df, running_df], axis=1)

    def transform_bills_features(self, df):
        recent3_paybills = [f"m{i}_paybill_total_value" for i in range(1, 4)]
        old3_paybills = [f"m{i}_paybill_total_value" for i in range(4, 7)]
        all_paybills = recent3_paybills + old3_paybills
        all_paybill_high_amount = [f"m{i}_paybill_highest_amount" for i in range(1, 7)]

        running_df = pd.DataFrame()
        running_df["paybill_hghamt_6m"] = df[all_paybill_high_amount].mean(axis=1)

        # Calculate coefficient of variance
        running_df["paybill_cv_3mrecent"] = df[recent3_paybills].std(axis=1) / df[recent3_paybills].mean(axis=1)
        running_df["paybill_cv_3mold"] = df[old3_paybills].std(axis=1) / df[old3_paybills].mean(axis=1)
        running_df["paybill_cv_6m"] = df[all_paybills].std(axis=1) / df[all_paybills].mean(axis=1)

        # Central tendency by median
        running_df["paybill_median_3mrecent"] = df[recent3_paybills].median(axis=1)
        running_df["paybill_median_6m"] = df[all_paybills].median(axis=1)

        # Paybill Volume calculations
        # Mean of recent 3 months and Mean of all 6 months
        running_df["paybill_vol_3mrecent"] = df[recent3_paybills].mean(axis=1)
        running_df["paybill_vol_6m"] = df[all_paybills].mean(axis=1)

        # Company Paybill Volume calculations vs total paybill volume and total amount spent on bill (total value)
        # recent3_volume = [f"m{i}_paybill_companies" for i in range(1, 4)]
        # old3_volume = [f"m{i}_paybill_companies" for i in range(4, 7)]
        all_volume = [f"m{i}_paybill_companies" for i in range(1, 7)]
        # df["paybill_vol_3mrecent"] = df[recent3_volume].mean(axis=1)
        # df["paybill_vol_6m"] = df[old3_volume].mean(axis=1)
        running_df["paybill_vol_max"] = df[all_volume].max(axis=1)
        # df["paybill_vol_min"] = df[all_volume].min(axis=1)

        # Their coeffience of covarience
        # df["paybill_cv_3mrecent"] = df[recent3_volume].std(axis=1) / df[recent3_volume].mean(axis=1)
        # df["paybill_cv_3mold"] = df[old3_volume].std(axis=1) / df[old3_volume].mean(axis=1)
        # df["paybill_cv_6m"] = df[all_volume].std(axis=1) / df[all_volume].mean(axis=1)

        # Find, averages and remove m{i}_bill_volume features
        bill_vol_features = [f"m{i}_paybill_volume" for i in range(1, 7)]
        running_df["bill_vol_avg"] = df[bill_vol_features].mean()

        # Drop all used raw features (without removing them model was slightly better than others)
        df.drop(columns=all_paybills, inplace=True)
        df.drop(columns=all_volume, inplace=True)
        df.drop(columns=all_paybill_high_amount, inplace=True)
        df.drop(columns=bill_vol_features, inplace=True)

        # Join two dfs
        df = pd.concat([df, running_df], axis=1)
        return df

    def transform_agent_features(self, df):
        # Deposit agents calculations
        recent3_deposit = [f"m{i}_deposit_agents" for i in range(1, 4)]
        old3_deposit = [f"m{i}_deposit_agents" for i in range(4, 7)]
        all_deposit = [f"m{i}_deposit_agents" for i in range(1, 7)]

        recent3_deposit_high_amount = [f"m{i}_deposit_highest_amount" for i in range(1, 4)]
        old3_deposit_high_amount = [f"m{i}_deposit_highest_amount" for i in range(4, 7)]
        all_deposit_high_amount = [f"m{i}_deposit_highest_amount" for i in range(1, 7)]

        running_df = pd.DataFrame()
        running_df["deposit_agents_hghmt_all"] = df[all_deposit_high_amount].mean(axis=1)
        # High amount drifts
        running_df["deposit_agents_hghamt_3mrecent_drift"] = df[recent3_deposit_high_amount].mean(axis=1) / df[
            old3_deposit_high_amount
        ].mean(axis=1)

        # Their coeffience of covarience
        running_df["deposit_agents_cv_3mrecent"] = df[recent3_deposit].std(axis=1) / df[recent3_deposit].mean(axis=1)
        running_df["deposit_agents_cv_3mold"] = df[old3_deposit].std(axis=1) / df[old3_deposit].mean(axis=1)
        running_df["deposit_agents_cv_6m"] = df[all_deposit].std(axis=1) / df[all_deposit].mean(axis=1)
        # running_df["deposit_agents_6total"] = df[all_deposit].sum(axis=1)

        # Ratio calculations
        # running_df["recent3_deposit_agents_ratio"] = df[recent3_deposit].sum(axis=1) / df[all_deposit].sum(axis=1)
        # running_df["old3_deposit_agents_ratio"] = df[old3_deposit].sum(axis=1) / df[all_deposit].sum(axis=1)

        # Drop all used raw features (without removing them model was slightly better than others)
        df.drop(columns=all_deposit, inplace=True)
        df.drop(columns=old3_deposit_high_amount, inplace=True)

        # Withdrawal agents calculations
        recent3_withdrawal = [f"m{i}_withdraw_agents" for i in range(1, 4)]
        old3_withdrawal = [f"m{i}_withdraw_agents" for i in range(4, 7)]
        all_withdrawal = [f"m{i}_withdraw_agents" for i in range(1, 7)]

        # recent3_withdraw_high_amount = [f"m{i}_withdraw_highest_amount" for i in range(1, 4)]
        old3_withdraw_high_amount = [f"m{i}_withdraw_highest_amount" for i in range(4, 7)]
        all_withdraw_high_amount = [f"m{i}_withdraw_highest_amount" for i in range(1, 7)]

        # running_df["withdraw_agents_hghamt_3mrecent"] = df[recent3_withdraw_high_amount].mean(axis=1)
        running_df["withdraw_agents_hghamt_6m"] = df[old3_withdraw_high_amount].mean(axis=1)
        # running_df["withdraw_agents_hghmt_all"] = df[all_withdraw_high_amount].mean(axis=1)
        running_df["agent_withdraw_min_max"] = df[all_withdraw_high_amount].max(axis=1) - df[
            all_withdraw_high_amount
        ].min(axis=1)

        # Their coeffience of covarience
        running_df["withdrawal_agents_cv_3mrecent"] = df[recent3_withdrawal].std(axis=1) / df[recent3_withdrawal].mean(
            axis=1
        )
        running_df["withdrawal_agents_cv_3mold"] = df[old3_withdrawal].std(axis=1) / df[old3_withdrawal].mean(axis=1)
        running_df["withdrawal_agents_cv_6m"] = df[all_withdrawal].std(axis=1) / df[all_withdrawal].mean(axis=1)
        # running_df["withdrawal_agents_6total"] = df[all_withdrawal].sum(axis=1)

        # Ratio calculations
        running_df["recent3_withdraw_agents_ratio"] = df[recent3_withdrawal].sum(axis=1) / df[all_withdrawal].sum(
            axis=1
        )
        running_df["old3_withdraw_agents_ratio"] = df[old3_withdrawal].sum(axis=1) / df[all_withdrawal].sum(axis=1)

        # Highest amount covariance calculations only
        running_df["withdrawal_agents_cv_3mrecent"] = df[recent3_withdrawal].std(axis=1) / df[recent3_withdrawal].mean(
            axis=1
        )
        running_df["withdrawal_agents_cv_3mold"] = df[old3_withdrawal].std(axis=1) / df[old3_withdrawal].mean(axis=1)
        running_df["withdrawal_agents_cv_6m"] = df[all_withdrawal].std(axis=1) / df[all_withdrawal].mean(axis=1)

        # Find, averages and drop m{i}_deposit_volume feature
        agent_vol_features = [f"m{i}_deposit_volume" for i in range(1, 7)]
        running_df["agent_vol_avg"] = df[agent_vol_features].mean(axis=1)
        # Drop all used raw features (without removing them model was slightly better than others)
        df.drop(columns=all_withdrawal, inplace=True)
        df.drop(columns=old3_withdraw_high_amount, inplace=True)
        df.drop(columns=agent_vol_features, inplace=True)

        # Join two dfs
        df = pd.concat([df, running_df], axis=1)

        return df

    def transform_mm_send_features(self, df: pd.DataFrame):
        # This is engineering is going to focus on range, max, min, and mean
        # Because I believe that differences can show what user is facing than variations
        mm_send_tot_value_recent3 = [f"m{i}_mm_send_total_value" for i in range(1, 4)]
        mm_send_tot_value_old3 = [f"m{i}_mm_send_total_value" for i in range(4, 7)]
        mm_send_tot_value_all = mm_send_tot_value_recent3 + mm_send_tot_value_old3
        mm_send_high_amount_recent3 = [f"m{i}_mm_send_highest_amount" for i in range(1, 4)]
        mm_send_high_amount_old3 = [f"m{i}_mm_send_highest_amount" for i in range(4, 7)]
        mm_send_high_amount_all = mm_send_high_amount_recent3 + mm_send_high_amount_old3
        mm_send_volume_recent3 = [f"m{i}_mm_send_volume" for i in range(1, 4)]
        mm_send_volume_old3 = [f"m{i}_mm_send_volume" for i in range(4, 7)]
        mm_send_volume_all = mm_send_volume_recent3 + mm_send_volume_old3

        running_df = pd.DataFrame()
        # Volatility
        running_df["mm_send_cv"] = df[mm_send_tot_value_all].std(axis=1) / df[mm_send_tot_value_all].mean(axis=1)
        running_df["max_min_ratio"] = df[mm_send_tot_value_all].max(axis=1) / df[mm_send_tot_value_all].min(axis=1)
        running_df["mmsend_3m_velocity"] = df[mm_send_tot_value_old3].mean(axis=1) / df[mm_send_tot_value_recent3].mean(
            axis=1
        )
        old_5months = [f"m{i}_mm_send_total_value" for i in reversed(range(2, 7))]
        running_df["historical_avg"] = df[old_5months] / df["m1_mm_send_total_value"]
        running_df["historical_z_score"] = (df["m1_mm_send_total_value"] - df[old_5months].mean(axis=1)) / df[
            old_5months
        ].std(axis=1)

        def find_monthovermonth(df: pd.DataFrame):
            new_df = pd.DataFrame()
            cols_copy = df.columns.to_list().copy()
            cols_copy.sort()

            for col in cols_copy:
                col_month = int(col[1])
                next_col = col_month + 1
                diff = df[col] - df[f"m{next_col}_{col[3:]}"]
                if next_col > 3:
                    new_df["m3_mm_send_tot_mom"] = diff
                    break
                else:
                    new_df[f"m{col_month}_mm_send_tot_mom"] = diff

            return new_df

        mom_recent3 = find_monthovermonth(df[[f"m{i}_mm_send_total_value" for i in range(1, 5)]])
        running_df[["m1_mm_send_tot_mom", "m2_mm_send_tot_mom", "m3_mm_send_tot_mom"]] = mom_recent3

        # Rename last_3months_month_over_month columns and add to running_df
        # last_3months_month_over_month.columns = [f"m{i}_mm_send_tot_mom" for i in range(4, 7)]
        # running_df = pd.concat([running_df, last_3months_month_over_month], axis=1)

        running_df["send_highamt_max_min_ratio"] = df[mm_send_high_amount_all].max(axis=1) / df[
            mm_send_high_amount_all
        ].min(axis=1)
        # running_df["send_highamt_max"] = df[mm_send_high_amount_all].max(axis=1)
        running_df["send_highamt_mean"] = df[mm_send_high_amount_all].median(axis=1)
        running_df["send_vol_mean"] = df[mm_send_volume_all].mean(axis=1)

        # try to understand the proportion of money send based on the money recieved
        # Drop all used raw features (without removing them model was slightly better than others)
        df.drop(columns=mm_send_tot_value_all, inplace=True)
        df.drop(columns=mm_send_high_amount_all, inplace=True)
        df.drop(columns=mm_send_volume_all, inplace=True)

        # Join two dfs
        df = pd.concat([df, running_df], axis=1)

        return df

    def month_over_month_calculation(self, field_suffix: str, df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
        regex = rf"^m\d+_[a-zA-Z0-9]+_({field_suffix})$"
        matched_cols = [col for col in df.columns if re.search(regex, col)]
        new_df_dict = {}

        for col in matched_cols:
            month = col[1]
            if int(month) == 6:
                # There is no change needed because it is the first month
                new_df_dict[f"{col}_mom"] = df[col]
            else:
                # Reverse the month numbering to get the preceeding month (M6->M1)
                preceeding_col = int(month) + 1
                col_suffix = col[3:]
                # Use np.where to avoid division by zero which results into np.inf which boosting models cannot handle
                change = df[col] - df[f"m{preceeding_col}_{col_suffix}"]
                new_df_dict[f"{col}_mom"] = change

                # Percentage change
                new_df_dict[f"{col}_mom"] = np.where(
                    df[f"m{preceeding_col}_{col_suffix}"] == 0,
                    change / 1,
                    change / df[f"m{preceeding_col}_{col_suffix}"],
                )

        return pd.DataFrame(new_df_dict), matched_cols

    def dail_average_balance_tranformation(self, df: pd.DataFrame):
        # Leave recent months in dataframe, aggregates, and remove old months
        recent3_months = [f"m{i}_daily_avg_bal" for i in range(1, 4)]
        old3_months = [f"m{i}_daily_avg_bal" for i in range(4, 7)]
        all_bal_cols = [f"m{i}_daily_avg_bal" for i in range(1, 7)]

        running_df = pd.DataFrame()
        # df["recent3_avg_balance"] = df[recent3_months].mean(axis=1)
        running_df["old3_avg_balance"] = df[old3_months].mean(axis=1)

        # Variation in balance using coefficient of variance
        running_df["bal_cv_3m"] = df[recent3_months].std(axis=1) / df[recent3_months].mean(axis=1)
        running_df["bal_cv_6m"] = df[all_bal_cols].std(axis=1) / df[all_bal_cols].mean(axis=1)
        running_df["bal_drift"] = (df[recent3_months].mean() + 0.1) / (df[old3_months].mean() + 0.1)

        # Remove old months
        df.drop(columns=old3_months, inplace=True)

        # Join two dfs
        df = pd.concat([df, running_df], axis=1)
        return df

    def transform(self, df: pd.DataFrame, is_train: bool = True) -> tuple[pd.DataFrame, pd.Series | None]:
        data = df.copy()
        # epsilon = 1e-6
        epsilon = 1

        # 1. Pop Target and drop ID
        y = data.pop(self.target_col) if (is_train and self.target_col in data) else None
        if self.id_col in data.columns:
            data.drop(columns=[self.id_col], inplace=True)

        # 2. Compute Consolidated Inflows and Outflows per Month (M1 to M6)
        inflow_channels = ["deposit", "received", "transfer_from_bank"]
        outflow_channels = ["withdraw", "paybill", "merchantpay", "mm_send"]

        new_features = {}
        for m in range(1, 7):
            # Sum monthly total values
            inf_cols = [f"m{m}_{ch}_total_value" for ch in inflow_channels if f"m{m}_{ch}_total_value" in data.columns]
            out_cols = [f"m{m}_{ch}_total_value" for ch in outflow_channels if f"m{m}_{ch}_total_value" in data.columns]

            # 2. Assign calculations to the dictionary instead of 'data'
            total_inflow = data[inf_cols].sum(axis=1) if inf_cols else 0.0
            total_outflow = data[out_cols].sum(axis=1) if out_cols else 0.0

            new_features[f"m{m}_total_inflow"] = total_inflow
            new_features[f"m{m}_total_outflow"] = total_outflow
            new_features[f"m{m}_net_flow"] = total_inflow - total_outflow

        # 3. Convert all new columns at once and join them horizontally
        new_df = pd.DataFrame(new_features, index=data.index)
        # Find any duplicates columns and remove them, but remain with the first one
        new_df = new_df.loc[:, ~new_df.columns.duplicated(keep="first")]
        data = pd.concat([data, new_df], axis=1)

        # 3. 3-Month Window Median Aggregations
        m1_3_inflow = [f"m{i}_total_inflow" for i in range(1, 4)]
        m4_6_inflow = [f"m{i}_total_inflow" for i in range(4, 7)]
        m1_3_outflow = [f"m{i}_total_outflow" for i in range(1, 4)]
        m4_6_outflow = [f"m{i}_total_outflow" for i in range(4, 7)]

        inflow_avg_recent_3m = data[m1_3_inflow].mean(axis=1)
        inflow_avg_baseline_3m = data[m4_6_inflow].mean(axis=1)
        outflow_avg_recent_3m = data[m1_3_outflow].mean(axis=1)
        outflow_avg_baseline_3m = data[m4_6_outflow].mean(axis=1)

        # 3M Flow Dynamics / Ratios
        data["inflow_drift_3m"] = (inflow_avg_recent_3m + epsilon) / (inflow_avg_baseline_3m + epsilon)
        data["outflow_drift_3m"] = (outflow_avg_recent_3m + epsilon) / (outflow_avg_baseline_3m + epsilon)
        data["net_coverage_recent_3m"] = (inflow_avg_recent_3m + epsilon) / (outflow_avg_recent_3m + epsilon)

        # 4. Slopes (3m and 6m Trajectories)
        # slope_df = pd.DataFrame()
        df["bal_slope_6m"] = self._compute_slopes(data, "daily_avg_bal", months=6)
        df["bal_slope_3m"] = self._compute_slopes(data, "daily_avg_bal", months=3)
        df["inflow_slope_6m"] = self._compute_slopes(data, "total_inflow", months=6)
        df["outflow_slope_6m"] = self._compute_slopes(data, "total_outflow", months=6)
        # print("Slope df shape: ", slope_df.shape)

        # 5. Granular End-Points (M1 & M6) and Ratios
        if "m1_daily_avg_bal" in data.columns and "m6_daily_avg_bal" in data.columns:
            data["bal_m1_m6_ratio"] = (data["m1_daily_avg_bal"] + epsilon) / (data["m6_daily_avg_bal"] + epsilon)

        if "m1_total_outflow" in data.columns and "m1_daily_avg_bal" in data.columns:
            data["m1_cashout_intensity"] = data["m1_total_outflow"] / (data["m1_daily_avg_bal"] + epsilon)

        # Last three months cashout intensity
        data["avg_recent3_cashout"] = data[m1_3_outflow].mean(axis=1)
        data["avg_old3_cashout"] = data[m4_6_outflow].mean(axis=1)

        # Variation in inflows using coefficient of variance
        data["inflow_cv_3m"] = data[m1_3_inflow].std(axis=1) / data[m1_3_inflow].mean(axis=1)
        data["inflow_cv_6m"] = data[m4_6_inflow].std(axis=1) / data[m4_6_inflow].mean(axis=1)

        # Variation in outflows using coefficient of variance
        data["outflow_cv_3m"] = data[m1_3_outflow].std(axis=1) / data[m1_3_outflow].mean(axis=1)
        data["outflow_cv_6m"] = data[m4_6_outflow].std(axis=1) / data[m4_6_outflow].mean(axis=1)

        # After m{i}_total_inflow/outflow calculations, remove them because the net_flow and the inflow/outflow_avg explains same thing
        data = data.drop(m4_6_inflow + m4_6_outflow, axis=1)

        # Do some aggregation on bills cols
        data = self.transform_bills_features(data)
        data = self.transform_agent_features(data)
        data = self.transform_merchant_features(data)
        data = self.transform_mm_send_features(data)
        # Run this modification afterall because it removes old 3 months
        data = self.dail_average_balance_tranformation(data)

        # Concatenate it horizontally with your original data
        data = pd.concat([data, new_df], axis=1)
        # data = pd.concat([data, slope_df], axis=1)

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

    def transform_for_test_set(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
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

    def split_dataset(self, raw_df, cross_validation=False, val_size=0.2, random_state=42):
        print(f"Submitted Dataset Shape: {raw_df.shape}, Val Size: {val_size}")
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

        print(X_train.shape, y_train.shape)
        # Currently best performer - Sept 2
        param_grid = {
            # 1. Direct control over positive class weight (scale down to boost precision)
            "scale_pos_weight": [
                # pos_weight * 0.5,
                # pos_weight * 0.75,
                # pos_weight,
                2.8333333333333335
            ],
            # 2. Tree structural constraints (controls overfitting/false positives)
            "max_depth": [7],
            # "max_depth": [6],
            "num_leaves": [15],
            # "min_data_in_leaf": [50],
            # 3. Regularization & Subsampling
            "colsample_bytree": [0.5],
            "reg_alpha": [10],
            "reg_lambda": [20],
            "learning_rate": [0.05],
            # "n_estimators": [400],
            "n_estimators": [1000],
        }

        lgb = LGBMClassifier(
            random_state=random_state,
            verbosity=-1,
            device=os.environ["device"],
            n_jobs=1,
            metric="logloss",
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
            "scale_pos_weight": [pos_weight],
            "max_depth": [7],
            "learning_rate": [0.05],
            "min_child_weight": [20],
            "objective": ["binary:logistic"],
            "colsample_bytree": [0.7],
            "subsample": [0.8],
            "reg_alpha": [10],
            "reg_lambda": [20],
        }

        estimator = XGBClassifier(
            n_jobs=1,
            random_state=random_state,
            device=os.environ["device"],
            eval_metric="aucpr",
            early_stopping_rounds=10,
        )

        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state)

        # Target F1 or Average Precision (PR-AUC) instead of generic accuracy
        grid = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="average_precision",  # Or 'average_precision' / 'roc_auc' / 'log_loss'
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
        # lgb_val_probs = self.xgb_model.predict_proba(X_val)[:, 1]
        lgb_val_probs = self.lgb_model.predict_proba(X_val)[:, 1]
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
        csv_path: str = "",
        raw_df: pd.DataFrame = None,
        val_size: float = 0.2,
        random_state: int = 42,
        threshold: float = 0.5,
        version: int = 1,
    ):
        """Loads dataset, performs stratified train_test_split, trains ensemble models, and prints validation metrics."""
        print(f"Reading dataset from: {csv_path}")
        if not csv_path and raw_df.empty:
            raise ValueError("Either csv_path or raw_df must be provided.")

        if csv_path and raw_df is None:
            raw_df = pd.read_csv(csv_path)

        X_train, y_train, X_val, y_val, pos_weight = self.split_dataset(
            raw_df, val_size=val_size, random_state=random_state
        )

        print(f"Features generated: {X_train.shape[1]} columns. Imbalance Ratio: {pos_weight:.2f}")

        # 3. Train LightGBM
        # self.fit_lgb(X_train, y_train, pos_weight, random_state)

        self.tune_lgbm(X_train, y_train, random_state)
        # self.tune_xgbost(X_train, y_train, X_val, y_val, random_state)
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
        train_pred, raw_train_pred = self.run_predictions(X_train, is_train=True, threshold=threshold)
        training_auc = roc_auc_score(y_train, raw_train_pred)
        print(f"Training ROC-AUC Score: {training_auc:.6f}")
        print("\nClassification Report:")
        print(classification_report(y_train, train_pred))
        print("====================================================\n\n")

        print(f"Log loss: {log_loss(y_val, raw_probs)}")
        print("====================================================\n\n")

        # 7. Save model pipeline
        filename = f"training_metrics_v{version}_f{X_train.shape[1]}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(
                {
                    "X_train": X_train,
                    "y_train": y_train,
                    "X_val": X_val,
                    "y_val": y_val,
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
        new_model_filename = f"model_pipeline_v{version}_f{num_features}.txt"
        # new_model_filename = f"model_pipeline_v{version}_f{num_features}.json"
        self.lgb_model.booster_.save_model(new_model_filename)
        # self.xgb_model.save_model(new_model_filename)
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
