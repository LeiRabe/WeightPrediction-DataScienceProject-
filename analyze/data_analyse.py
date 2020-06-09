import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

# quick views
df.plot(kind='scatter', x='Weight', y='Height', color='blue')
plt.show()

gender_colors = {
    "Male": "r",
    "Female": "g"
}

ax = plt.subplot()
for gender in ["Male", "Female"]:
    color = gender_colors[gender]
    df[df.Gender == gender].plot(kind='scatter', x='Weight', y='Height', label=gender, ax=ax, color=color)
handles, labels = ax.get_legend_handles_labels()
_ = ax.legend(handles, labels, loc="upper left")

plt.show()

