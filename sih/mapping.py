# type: ignore
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import mplcursors as mc

main_dir = Path(__file__).parent

csv_path = main_dir / "Synchronised V abd S datasets" / "Categorised IOVNB Dataset" / "S (Driver A)" / "S1" / "S-S1.csv"

df = pd.read_csv(csv_path, encoding='latin1')

xtime = df.iloc[:, 7].values
xtime = xtime / 1000
ykmh  = df.iloc[:, 3].values
yaccx = df.iloc[:, 9].values
yaccy = df.iloc[:, 10].values
yaccz = df.iloc[:, 11].values

plt.xlabel("Time in seconds")
plt.ylabel("Car speed and sensor metrics")

plt.plot(xtime[:1000], ykmh[:1000], label = "Speed(km/h)")
plt.plot(xtime[:1000], yaccx[:1000], label = "Acc X")
plt.plot(xtime[:1000], yaccy[:1000], label = "Acc Y")
plt.plot(xtime[:1000], yaccz[:1000], label = "Acc Z")

# plt.plot(xtime, ykmh, label = "Speed(km/h)")
# plt.plot(xtime, yaccx, label = "Acc X")
# plt.plot(xtime, yaccy, label = "Acc Y")
# plt.plot(xtime, yaccz, label = "Acc Z")


plt.legend()

mc.cursor(hover=True)

plt.show()