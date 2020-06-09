import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("datasets_26073_33239_weight-height.csv")
# inches to cm
height = df["Height"].tolist()
height_cm = []
for h in height:
    h *= 2.54
    height_cm.append(h)

df["Height"] = height_cm

# lbs to kg
weight = df["Weight"].tolist()
weight_kg = []
for w in weight:
    w *= 0.453592
    weight_kg.append(w)

df["Weight"] = weight_kg

# quick view
# df.plot(kind='scatter',x='Weight',y='Height',color='red')
# plt.show()



