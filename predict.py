import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import argparse

def add_lags(df, col):
    out = df.copy()
    for lag in [1, 2, 3, 6,12]:
        out[f"lag_{lag}"] = out[col].shift(lag)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="challenge_data")
    parser.add_argument("--output", type=str, default="predictions.csv")
    args = parser.parse_args()

    # Read CPU data
    gz_file = os.path.join(args.data, "vm_cpu_readings.csv.gz")
    csv_file = os.path.join(args.data, "vm_cpu_readings.csv")

    if os.path.exists(gz_file):
        df_cpu = pd.read_csv(gz_file)
    elif os.path.exists(csv_file):
        df_cpu = pd.read_csv(csv_file)
    else:
        raise FileNotFoundError(f"No cpu readings file found in {args.data}")

    # Aggregate 5-minute intervals
    df_agg = (
        df_cpu.groupby("timestamp", as_index=False)
        .agg(
            active_vm_count=("vm_id", "nunique"),
            aggregate_cpu_utilization=("avg_cpu", "mean"),
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Keep final 24 timestamps as holdout
    holdout_len = 24
    train_df = df_agg.iloc[:-holdout_len].copy().reset_index(drop=True)
    holdout_ts = df_agg["timestamp"].iloc[-holdout_len:].tolist()

    features = ["lag_1", "lag_2", "lag_3", "lag_6","lag_12"]

    # Train models on training split only
    cpu_train = add_lags(train_df, "aggregate_cpu_utilization").dropna()
    model_cpu = LinearRegression()
    model_cpu.fit(cpu_train[features], cpu_train["aggregate_cpu_utilization"])

    vm_train = add_lags(train_df, "active_vm_count").dropna()
    model_vm = LinearRegression()
    model_vm.fit(vm_train[features], vm_train["active_vm_count"])

    # Recursive prediction for the 24 holdout steps
    cpu_hist = train_df["aggregate_cpu_utilization"].tolist()
    vm_hist = train_df["active_vm_count"].tolist()
    rows = []

    for ts in holdout_ts:
        x_cpu = pd.DataFrame(
            [[cpu_hist[-1], cpu_hist[-2], cpu_hist[-3], cpu_hist[-6],cpu_hist[-12]]],
            columns=features,
        )
        x_vm = pd.DataFrame(
            [[vm_hist[-1], vm_hist[-2], vm_hist[-3], vm_hist[-6], vm_hist[-12]]],
            columns=features,
        )

        pred_cpu = float(np.clip(model_cpu.predict(x_cpu)[0], 0.0, 100.0))
        pred_vm = int(max(0, round(float(model_vm.predict(x_vm)[0]))))

        rows.append({
            "timestamp": ts,
            "active_vm_count": pred_vm,
            "aggregate_cpu_utilization": pred_cpu,
        })

        cpu_hist.append(pred_cpu)
        vm_hist.append(pred_vm)

    res = pd.DataFrame(rows)
    res.to_csv(args.output, index=False)
    print(f"Saved {len(res)} rows to {args.output}")


if __name__ == "__main__":
    main()