import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,root_mean_squared_error,mean_absolute_percentage_error
from datetime import timedelta

def compute_metrics(y_true,y_pred)->dict:
    rmse=root_mean_squared_error(y_true=y_true,y_pred=y_pred)
    mae=mean_absolute_error(y_true=y_true,y_pred=y_pred)
    mask=y_true!=0
    mape=mean_absolute_percentage_error(y_true=y_true[mask],y_pred=y_pred[mask])

    return {"rmse":rmse,"mae":mae,"mape":mape}

def plot_predictions(
    val_df: pl.DataFrame,
    y_pred: np.ndarray,
    zone_id: int,
    title: str,
    save_path: Path,
) -> Path:

    pred_df = val_df.with_columns(
        pl.Series("prediction", y_pred)
    )

    zone_df = (
        pred_df
        .filter(pl.col("PULocationID") == zone_id)
        .sort("pickup_hour")
        .head(168)
    )

    if zone_df.is_empty():
        raise ValueError(
            f"No validation data found for zone {zone_id}"
        )

    hours = zone_df["pickup_hour"].to_list()
    y_true = zone_df["trip_count"].to_numpy()
    y_pred_zone = zone_df["prediction"].to_numpy()

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        hours,
        y_true,
        label="Actual",
        linewidth=2,
    )

    ax.plot(
        hours,
        y_pred_zone,
        label="Predicted",
        linewidth=2,
        linestyle="--",
    )

    rush_hours = {7, 8, 9, 17, 18, 19}

    for dt in hours:
        if dt.hour in rush_hours:
            ax.axvspan(
                dt,
                dt + timedelta(hours=1),
                alpha=0.08,
            )

    ax.set_title(
        f"{title} | Zone {zone_id}"
    )

    ax.set_xlabel("Pickup Hour")
    ax.set_ylabel("Trip Count")

    ax.legend()
    ax.grid(alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()

    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    return save_path
