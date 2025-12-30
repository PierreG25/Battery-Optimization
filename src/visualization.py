import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


csv_path = 'data/dispatch_results.csv'
# csv_path = 'data/clean_spot_price_01:11:2025.csv'

df = pd.read_csv(csv_path)

# plt.step(df['MTU (CET/CEST)'], df['Day-ahead Price (EUR/MWh)'], where='post', label='Price (EUR/MWh)')
# plt.xlabel('Time (hours)')
# plt.ylabel('Price (EUR/MWh)')
# plt.grid(True, alpha=0.3)
# plt.show()

plt.step(df['t'], df['price'], label='Price (EUR/MWh)')
plt.xlabel('Time (hours)')
plt.ylabel('Price (EUR/MWh)')
plt.grid(True, alpha=0.3)
plt.show()

x = df['t']
y1 = df['price']
y2 = df['soc_MWh_start']
fig, ax1 = plt.subplots()

ax1.step(x, y1, where="post")
ax1.set_ylabel("Price EUR/MWh")

ax2 = ax1.twinx()
ax2.step(x, y2, where="post", color='orange', linestyle='--')
ax2.set_ylabel("SOC")

ax1.set_xlabel("Time")
plt.title("Decision vs demand")
plt.show()
