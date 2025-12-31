"""
battery_arbitrage_pyomo_multi_day.py

Run the single-day Pyomo battery arbitrage optimization independently
for each day in one dataset (CSV with many consecutive days).

Input CSV columns:
- time
- price
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import pyomo.environ as pyo


@dataclass(frozen=True)
class BatteryParams:
    # Energy capacity (MWh)
    e_max: float = 10.0
    e_min: float = 0.0

    # Power limits (MW)
    p_ch_max: float = 5.0
    p_dis_max: float = 5.0

    # Efficiencies
    eta_ch: float = 0.95
    eta_dis: float = 0.95

    # Initial SOC (MWh) (RESET EACH DAY for independent optimization)
    soc0: float = 5.0

    # Optional terminal SOC requirement (MWh)
    soc_terminal: Optional[float] = 5.0


def read_dataset(csv_path: str) -> tuple[pd.DataFrame, float]:
    """
    Reads the full dataset and returns:
    - df with columns: time (datetime), price (float), date (python date)
    - inferred timestep length dt_hours (float)
    """
    df = pd.read_csv(csv_path)

    if "time" not in df.columns:
        raise ValueError("CSV must contain a 'time' column.")
    if "price" not in df.columns:
        raise ValueError("CSV must contain 'price' column.")

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    if len(df) < 2:
        raise ValueError("Need at least two rows to infer timestep.")

    # Robust timestep inference: use the most common delta (mode)
    deltas = df["time"].diff().dropna()
    dt_seconds = deltas.dt.total_seconds().round().mode().iloc[0]
    dt_hours = float(dt_seconds) / 3600.0

    df["date"] = df["time"].dt.date
    return df, dt_hours


def build_and_solve_one_day(
    prices: pd.Series,
    dt_hours: float,
    params: BatteryParams,
    solver_name: str = "highs",
):
    """
    Your single-day optimization (unchanged in spirit).
    """
    T = len(prices)

    m = pyo.ConcreteModel("BatteryArbitrageDay")

    # Time sets
    m.T = pyo.RangeSet(0, T - 1)
    m.S = pyo.RangeSet(0, T)

    # Prices
    price_dict = {t: float(prices.iloc[t]) for t in range(T)}
    m.price = pyo.Param(m.T, initialize=price_dict)

    # Decision variables
    m.p_ch = pyo.Var(m.T, domain=pyo.NonNegativeReals)   # MW
    m.p_dis = pyo.Var(m.T, domain=pyo.NonNegativeReals) # MW
    m.soc = pyo.Var(m.S)                                # MWh

    # Bounds
    m.soc_bounds = pyo.Constraint(
        m.S, rule=lambda m, s: (params.e_min, m.soc[s], params.e_max)
    )
    m.pch_bounds = pyo.Constraint(
        m.T, rule=lambda m, t: m.p_ch[t] <= params.p_ch_max
    )
    m.pdis_bounds = pyo.Constraint(
        m.T, rule=lambda m, t: m.p_dis[t] <= params.p_dis_max
    )

    # Initial SOC (for independent days, always params.soc0)
    m.soc_init = pyo.Constraint(expr=m.soc[0] == params.soc0)

    # SOC dynamics
    def soc_dyn_rule(m, t):
        return (
            m.soc[t + 1]
            == m.soc[t]
            + (params.eta_ch * m.p_ch[t]
               - (1.0 / params.eta_dis) * m.p_dis[t]) * dt_hours
        )
    m.soc_dyn = pyo.Constraint(m.T, rule=soc_dyn_rule)

    # Terminal SOC constraint
    if params.soc_terminal is not None:
        m.soc_terminal = pyo.Constraint(expr=m.soc[T] >= params.soc_terminal)

    # Objective: maximize profit
    m.obj = pyo.Objective(
        expr=sum(
            m.price[t] * (m.p_dis[t] - m.p_ch[t]) * dt_hours
            for t in m.T
        ),
        sense=pyo.maximize,
    )

    # Solve
    solver = pyo.SolverFactory(solver_name)
    res = solver.solve(m, tee=False)

    # Collect results
    out = pd.DataFrame({
        "step": list(range(T)),
        "t_hours": [t * dt_hours for t in range(T)],
        "price": prices.values,
        "p_charge_MW": [pyo.value(m.p_ch[t]) for t in range(T)],
        "p_discharge_MW": [pyo.value(m.p_dis[t]) for t in range(T)],
        "soc_MWh_start": [pyo.value(m.soc[t]) for t in range(T)],
        "soc_MWh_end": [pyo.value(m.soc[t + 1]) for t in range(T)],
    })

    out["profit_EUR"] = out["price"] * (out["p_discharge_MW"] - out["p_charge_MW"]) * dt_hours

    # Daily KPIs
    charged_MWh = float((out["p_charge_MW"] * dt_hours).sum())
    discharged_MWh = float((out["p_discharge_MW"] * dt_hours).sum())
    throughput_MWh = charged_MWh + discharged_MWh
    full_cycles = throughput_MWh / (2.0 * params.e_max) if params.e_max > 0 else 0.0
    total_profit = float(out["profit_EUR"].sum())

    return m, res, out, total_profit, full_cycles


def run_independent_daily_optimizations(
    csv_path: str,
    params: BatteryParams,
    solver_name: str = "highs",
    skip_incomplete_days: bool = False,
):
    os.makedirs("data", exist_ok=True)

    df, dt_hours = read_dataset(csv_path)
    print(f"Inferred timestep: {dt_hours:.4f} hours")

    expected_points = int(round(24.0 / dt_hours))  # 96 for 15-min, 24 for hourly

    all_days_dispatch = []
    daily_summary_rows = []

    for day, g in df.groupby("date", sort=True):
        g = g.sort_values("time").reset_index(drop=True)

        if skip_incomplete_days and len(g) != expected_points:
            print(f"Skipping {day} (points={len(g)} expected={expected_points})")
            continue

        prices = g["price"]

        try:
            _, res, day_dispatch, day_profit, day_cycles = build_and_solve_one_day(
                prices=prices,
                dt_hours=dt_hours,
                params=params,
                solver_name=solver_name,
            )
        except Exception as e:
            print(f"Day {day}: failed ({e})")
            continue

        # Attach timestamps + identifiers
        day_dispatch.insert(0, "time", g["time"].values)
        day_dispatch.insert(1, "date", day)
        day_dispatch["soc_percent_start"] = day_dispatch["soc_MWh_start"] / params.e_max * 100.0

        all_days_dispatch.append(day_dispatch)

        daily_summary_rows.append({
            "date": day,
            "n_points": len(g),
            "dt_hours": dt_hours,
            "profit_EUR": day_profit,
            "full_cycles": day_cycles,
        })

    if not all_days_dispatch:
        raise RuntimeError("No days were optimized successfully (check input data).")

    dispatch_all = pd.concat(all_days_dispatch, ignore_index=True)
    daily_summary = pd.DataFrame(daily_summary_rows).sort_values("date").reset_index(drop=True)

    dispatch_all.to_csv("data/dispatch_results_all_days_15.csv", index=False)
    daily_summary.to_csv("data/daily_summary_15.csv", index=False)

    print("Wrote data/dispatch_results_all_days.csv")
    print("Wrote data/daily_summary.csv")
    print("\nTop 5 days by profit:")
    print(daily_summary.sort_values("profit_EUR", ascending=False).head(5))

    return dispatch_all, daily_summary


def main():
    if len(sys.argv) < 2:
        print("Usage: python battery_arbitrage_pyomo_multi_day.py path/to/prices.csv")
        sys.exit(1)

    csv_path = sys.argv[1]

    params = BatteryParams(
        e_max=10.0,
        e_min=0.0,
        p_ch_max=5.0,
        p_dis_max=5.0,
        eta_ch=0.95,
        eta_dis=0.95,
        soc0=5.0,
        soc_terminal=5.0,
    )

    run_independent_daily_optimizations(
        csv_path=csv_path,
        params=params,
        solver_name="highs",
        skip_incomplete_days=False,  # set True if you want only full days
    )


if __name__ == "__main__":
    main()
