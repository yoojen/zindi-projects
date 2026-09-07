import pandas as pd
import numpy as np


MONTHS = range(1, 7)

# channel_name -> (direction, column_prefix, counterparty_suffix)
CHANNEL_CONFIG = {
    "paybill": ("outflow", "paybill", "companies"),
    "merchantpay": ("outflow", "merchantpay", "merchants"),
    "mm_send": ("outflow", "mm_send", "recipients"),
    "withdraw": ("outflow", "withdraw", "agents"),
    "transfer_from_bank": ("inflow", "transfer_from_bank", "banks"),
    "received": ("inflow", "received", "senders"),
    "deposit": ("inflow", "deposit", "agents"),
}

OUTFLOW_CHANNELS = [c for c, (d, _, _) in CHANNEL_CONFIG.items() if d == "outflow"]
INFLOW_CHANNELS = [c for c, (d, _, _) in CHANNEL_CONFIG.items() if d == "inflow"]


def aggregate_channel_across_months(df: pd.DataFrame, channel: str) -> pd.DataFrame:
    """
    Collapse one channel's 6 monthly columns into whole-window features:
      - {channel}_total_volume        : sum of transaction counts, M1-M6
      - {channel}_total_value         : sum of transaction value, M1-M6
      - {channel}_avg_monthly_value   : mean monthly value (handles missing months)
      - {channel}_max_single_txn      : largest single transaction seen in the window
      - {channel}_avg_unique_cparty   : average unique counterparties per active month
      - {channel}_active_months       : number of months with volume > 0
      - {channel}_value_volatility    : coefficient of variation of monthly value
                                         (std/mean) -> flags erratic/spiking usage
    """
    _, prefix, cparty_suffix = CHANNEL_CONFIG[channel]

    vol_cols = [f"m{m}_{prefix}_volume" for m in MONTHS]
    val_cols = [f"m{m}_{prefix}_total_value" for m in MONTHS]
    high_cols = [f"m{m}_{prefix}_highest_amount" for m in MONTHS]
    cparty_cols = [f"m{m}_{prefix}_{cparty_suffix}" for m in MONTHS]

    vol = df[vol_cols].fillna(0)
    val = df[val_cols].fillna(0)
    high = df[high_cols].fillna(0)
    cparty = df[cparty_cols].fillna(0)

    out = pd.DataFrame(index=df.index)
    out[f"{channel}_total_volume"] = vol.sum(axis=1)
    out[f"{channel}_total_value"] = val.sum(axis=1)
    out[f"{channel}_avg_monthly_value"] = val.mean(axis=1)
    out[f"{channel}_max_single_txn"] = high.max(axis=1)
    out[f"{channel}_active_months"] = (vol > 0).sum(axis=1)
    out[f"{channel}_avg_unique_cparty"] = cparty.replace(0, np.nan).mean(axis=1).fillna(0)

    mean_val = val.mean(axis=1)
    std_val = val.std(axis=1)
    out[f"{channel}_value_volatility"] = (std_val / mean_val.replace(0, np.nan)).fillna(0)

    return out


def build_all_channel_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Run aggregate_channel_across_months for every channel and concat results."""
    pieces = [aggregate_channel_across_months(df, ch) for ch in CHANNEL_CONFIG]
    return pd.concat(pieces, axis=1)


def compute_channel_shares(agg_df: pd.DataFrame, direction: str = "outflow") -> pd.DataFrame:
    """
    Share of each channel's *value* within its direction's total (outflow or inflow),
    over the whole 6-month window. Requires agg_df from build_all_channel_aggregates.
    """
    channels = OUTFLOW_CHANNELS if direction == "outflow" else INFLOW_CHANNELS
    val_cols = [f"{ch}_total_value" for ch in channels]

    total = agg_df[val_cols].sum(axis=1)
    shares = pd.DataFrame(index=agg_df.index)
    for ch in channels:
        shares[f"{ch}_share_of_{direction}"] = (agg_df[f"{ch}_total_value"] / total.replace(0, np.nan)).fillna(0)

    shares[f"total_{direction}_value"] = total
    return shares


def compute_channel_diversity(agg_df: pd.DataFrame, direction: str = "outflow") -> pd.Series:
    """
    Shannon entropy of value-share across channels in one direction.
    Higher = spread evenly across channels. Lower = concentrated in one/two channels.
    Note: on its own this is ambiguous for stress -- pair it with WHICH channel
    dominates (see compute_channel_shares) rather than reading entropy alone.
    """
    shares = compute_channel_shares(agg_df, direction)
    share_cols = [c for c in shares.columns if c.endswith(f"_share_of_{direction}")]
    p = shares[share_cols].values
    p = np.clip(p, 1e-12, 1)  # avoid log(0)
    entropy = -(p * np.log(p)).sum(axis=1)
    return pd.Series(entropy, index=agg_df.index, name=f"{direction}_channel_entropy")


def compute_channel_share_trend(
    df: pd.DataFrame,
    channel: str,
    direction: str = "outflow",
    recent_months=(1, 2, 3),
    earlier_months=(4, 5, 6),
) -> pd.DataFrame:
    """
    Compares one channel's share of total outflow/inflow value in a recent window
    vs. an earlier window, using RAW monthly columns (not the 6-month aggregate).

    Returns:
      - {channel}_share_recent   : channel value / total {direction} value, recent months
      - {channel}_share_earlier  : same, earlier months
      - {channel}_share_trend    : recent - earlier
                                    (positive => growing reliance on this channel;
                                     for 'withdraw', positive = worsening liquidity signal)

    Example: compute_channel_share_trend(df, "withdraw") gives withdraw_share_trend,
    comparing M1-M3 vs M4-M6 by default.
    """
    _, prefix, _ = CHANNEL_CONFIG[channel]
    direction_channels = OUTFLOW_CHANNELS if direction == "outflow" else INFLOW_CHANNELS

    def period_share(months):
        chan_val_cols = [f"m{m}_{prefix}_total_value" for m in months]
        chan_val = df[chan_val_cols].fillna(0).sum(axis=1)

        total_val_cols = []
        for ch in direction_channels:
            _, ch_prefix, _ = CHANNEL_CONFIG[ch]
            total_val_cols += [f"m{m}_{ch_prefix}_total_value" for m in months]
        total_val = df[total_val_cols].fillna(0).sum(axis=1)

        return (chan_val / total_val.replace(0, np.nan)).fillna(0)

    recent_share = period_share(recent_months)
    earlier_share = period_share(earlier_months)

    out = pd.DataFrame(index=df.index)
    out[f"{channel}_share_recent"] = recent_share
    out[f"{channel}_share_earlier"] = earlier_share
    out[f"{channel}_share_trend"] = recent_share - earlier_share

    return out


def compute_liquidity_signals(df: pd.DataFrame, agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Higher-level stress-oriented features combining balance + flows.
    """
    sig = pd.DataFrame(index=df.index)

    bal_cols = [f"m{m}_daily_avg_bal" for m in MONTHS]
    bal = df[bal_cols].fillna(0)
    sig["avg_balance_6m"] = bal.mean(axis=1)
    sig["balance_trend"] = bal["m1_daily_avg_bal"] - bal["m6_daily_avg_bal"]  # recent - oldest
    sig["balance_volatility"] = (bal.std(axis=1) / bal.mean(axis=1).replace(0, np.nan)).fillna(0)

    out_shares = compute_channel_shares(agg_df, "outflow")
    in_shares = compute_channel_shares(agg_df, "inflow")

    sig["withdraw_share_of_outflow"] = out_shares["withdraw_share_of_outflow"]
    sig["paybill_share_of_outflow"] = out_shares["paybill_share_of_outflow"]
    sig["total_outflow_value"] = out_shares["total_outflow_value"]
    sig["total_inflow_value"] = in_shares["total_inflow_value"]
    sig["outflow_to_inflow_ratio"] = (sig["total_outflow_value"] / sig["total_inflow_value"].replace(0, np.nan)).fillna(0)

    sig["outflow_channel_entropy"] = compute_channel_diversity(agg_df, "outflow")
    sig["withdraw_active_months"] = agg_df["withdraw_active_months"]
    sig["paybill_active_months"] = agg_df["paybill_active_months"]

    withdraw_trend = compute_channel_share_trend(df, "withdraw", "outflow")
    sig["withdraw_share_trend"] = withdraw_trend["withdraw_share_trend"]

    paybill_trend = compute_channel_share_trend(df, "paybill", "outflow")
    sig["paybill_share_trend"] = paybill_trend["paybill_share_trend"]

    return sig
