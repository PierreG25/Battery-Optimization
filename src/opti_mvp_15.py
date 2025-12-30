"""
battery_arbitrage_pyomo_mvp.py

MVP: Optimize battery charge/discharge against spot prices.
- Linear program (LP)
- Variables: p_ch[t], p_dis[t], soc[t]
- Constraints: SOC dynamics, bounds, power limits
- Objective: maximize profit = sum(price * (dis - ch) * dt)

Supports arbitrary timesteps (e.g. 15 min, 30 min, 1h).

Input CSV columns:
- time
- price
"""

from __future__ import annotations

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

    # Initial SOC (MWh)
    soc0: float = 5.0

    # Optional terminal SOC requirement (MWh)
    soc_terminal: Optional[float] = 5.0


def read_prices(csv_path: str) -> tuple[pd.Series, float]:
    """
    Reads a CSV and returns:
    - prices (pd.Series)
    - timestep length in hours (float)
    """
    df = pd.read_csv(csv_path)

    if "time" not in df.columns:
        raise ValueError("CSV must contain a 'time' column.")
    if "price" not in df.columns:
        raise ValueError("CSV must contain 'price' column.")

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    prices = df["price"].astype(float)

    if len(prices) < 2:
        raise ValueError("Need at least two times to infer timestep.")

    # 🔹 Infer timestep in hours (e.g. 15 min → 0.25 h)
    dt_hours = (df["time"].iloc[1] - df["time"].iloc[0]).total_seconds() / 3600.0

    return prices, dt_hours


def build_and_solve(
    prices: pd.Series,
    dt_hours: float,
    params: BatteryParams,
    solver_name: str = "highs",
):
    """
    Builds and solves the Pyomo model.
    """
    T = len(prices)

    m = pyo.ConcreteModel("BatteryArbitrageMVP")

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

    # Initial SOC
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
        m.soc_terminal = pyo.Constraint(
            expr=m.soc[T] >= params.soc_terminal
        )

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
        "t": [t*dt_hours for t in range(T)],
        "price": prices.values,
        "p_charge_MW": [pyo.value(m.p_ch[t]) for t in range(T)],
        "p_discharge_MW": [pyo.value(m.p_dis[t]) for t in range(T)],
        "soc_MWh_start": [pyo.value(m.soc[t]) for t in range(T)],
        "soc_MWh_end": [pyo.value(m.soc[t + 1]) for t in range(T)],
        "full_cycles": sum((pyo.value(m.p_ch[t]) + pyo.value(m.p_dis[t])) * dt_hours for t in range(T))/(2*params.e_max),
    })

    out["profit_EUR"] = (
        out["price"] * (out["p_discharge_MW"] - out["p_charge_MW"]) * dt_hours
    )

    return m, res, out, out["profit_EUR"].sum()


def main():
    if len(sys.argv) < 2:
        print("Usage: python battery_arbitrage_pyomo_mvp.py path/to/prices.csv")
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

    prices, dt_hours = read_prices(csv_path)
    print(f"Inferred timestep: {dt_hours:.2f} hours")

    _, solver_results, results_df, total_profit = build_and_solve(
        prices, dt_hours, params
    )

    print("\n=== Solve status ===")
    print(solver_results.solver.status)
    print(solver_results.solver.termination_condition)
    print(f"\nTotal profit (EUR): {total_profit:.2f}")

    results_df["soc_percent"] = results_df["soc_MWh_start"] / params.e_max * 100.0
    print(results_df)
    results_df.to_csv("data/dispatch_results_15.csv", index=False)
    print("Wrote data/dispatch_results_15.csv")


if __name__ == "__main__":
    main()
