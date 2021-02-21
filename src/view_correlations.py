import os
import pandas as pd
from heatmap import heatmap, corrplot
from matplotlib import pyplot as plt
from pylab import rcParams

rcParams["figure.figsize"] = 7, 7
import seaborn as sns
import numpy as np


data = pd.read_csv(
    os.path.join(
        "..",
        "data",
        "Data",
        "Aspirational Data",
        "processed-aspirational_overall_2021_02_04_22-v0_1-4d1a82f.csv",
    )
)

subset_data = data[
    [
        "tdd_per_kg",
        "avg_total_insulin_per_day_outcomes",
        "avg_basal_insulin_per_day_outcomes",
        "total_daily_scheduled_basal",
        "avg_isf",
        "weighted_cir_outcomes",
        "avg_carbs_per_day_outcomes",
        "Age",
        "BMI",
        "weight",
    ]
]

corr = subset_data.corr("spearman")
corr.to_csv("spearman_correlation_all.csv")
fig = plt.figure(figsize=(6, 6))

corrplot(corr, size_scale=300)
ax = fig.gca()
ax.grid(False)
ax.set_xticks([])
plt.show()

data = data.rename(
    columns={
        "total_daily_scheduled_basal": "Basal",
        "avg_isf": "ISF",
        "weighted_cir_outcomes": "CIR",
        "avg_carbs_per_day_outcomes": "Carbs",
        "tdd_per_kg": "TDD",
    }
)

new_subset = data[["TDD", "Basal", "ISF", "CIR", "Carbs", "Age", "BMI",]]

corr = new_subset.corr("spearman")
corr.to_csv("spearman_correlation_reduced.csv")
fig = plt.figure(figsize=(6, 6))

corrplot(corr, size_scale=300)
plt.show()
