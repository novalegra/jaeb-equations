import pandas as pd
import utils
from pathlib import Path
import math
from equation_utils import *


input_file_name = "processed-aspirational_overall_2021_02_15_20-v0_1-9eea51f"
data_path = utils.find_full_path(input_file_name, ".csv")
should_save_graph = True

df = pd.read_csv(data_path)

tdd = "avg_total_insulin_per_day_outcomes"
basal = "total_daily_scheduled_basal"  # Total daily basal
total_daily_carbs = "avg_carbs_per_day_outcomes"  # Total daily CHO
isf = "avg_isf"
icr = "weighted_cir_outcomes"
tir = "percent_70_180"
percent_below_40 = "percent_below_40"
percent_below_54 = "percent_below_54"
percent_below_70 = "percent_below_70"
percent_above_180 = "percent_above_180"
percent_above_250 = "percent_above_250"
percent_above_400 = "percent_above_400"
percent_cgm = "percent_cgm_available"
days_insulin = None
loop_id = "loop_id"

# From the survey data
bmi = "BMI"
bmi_percentile = "BMIPercentile"
age = "Age"


def make_graph(y_true, y_predicted, axes_labels, title):
    residuals = (y_true - y_predicted) / y_predicted * 100
    utils.two_dimension_plot(
        y_true, residuals, axes_labels, title, [-200, 200], save=should_save_graph,
    )


""" Basal """
df["jaeb_predicted_basals"] = df.apply(
    lambda x: jaeb_basal_equation(x[tdd], x[total_daily_carbs]), axis=1
)
make_graph(
    df[basal],
    df["jaeb_predicted_basals"],
    ["Scheduled Basal (U/day)", "Residual (%)"],
    "Residuals for \nProposed Basal Equation",
)

df["traditional_fitted_predicted_basals"] = df.apply(
    lambda x: traditional_basal_equation(x[tdd]), axis=1
)
make_graph(
    df[basal],
    df["traditional_fitted_predicted_basals"],
    ["Scheduled Basal (U/day)", "Residual (%)"],
    "Residuals for \n Fitted ACE Basal Equation",
)

df["traditional_constants_predicted_basals"] = df.apply(
    lambda x: traditional_constants_basal_equation(x[tdd]), axis=1
)
make_graph(
    df[basal],
    df["traditional_constants_predicted_basals"],
    ["Scheduled Basal (U/day)", "Residual (%)"],
    "Residuals for \n Traditional ACE Basal Equation",
)


# """ ISF Analysis """
df["jaeb_predicted_isf"] = df.apply(lambda x: jaeb_isf_equation(x[tdd], x[bmi]), axis=1)
make_graph(
    df[isf],
    df["jaeb_predicted_isf"],
    ["Scheduled ISF (mg/dL/U)", "Residual (%)"],
    "Residuals for \nProposed ISF Equation",
)

df["traditional_fitted_predicted_isf"] = df.apply(
    lambda x: traditional_isf_equation(x[tdd]), axis=1
)
make_graph(
    df[isf],
    df["traditional_fitted_predicted_isf"],
    ["Scheduled ISF (mg/dL/U)", "Residual (%)"],
    "Residuals for \n Fitted ACE ISF Equation",
)

df["traditional_constants_predicted_isf"] = df.apply(
    lambda x: traditional_constants_isf_equation(x[tdd]), axis=1
)
make_graph(
    df[isf],
    df["traditional_constants_predicted_isf"],
    ["Scheduled ISF (mg/dL/U)", "Residual (%)"],
    "Residuals for \nTraditional ACE ISF Equation",
)

""" ICR Analysis """
df["jaeb_predicted_icr"] = df.apply(
    lambda x: jaeb_icr_equation(x[tdd], x[total_daily_carbs]), axis=1
)
make_graph(
    df[icr],
    df["jaeb_predicted_icr"],
    ["Scheduled CIR (g/U)", "Residual (%)"],
    "Residuals for \nProposed CIR Equation",
)

df["traditional_fitted_predicted_icr"] = df.apply(
    lambda x: traditional_icr_equation(x[tdd]), axis=1
)
make_graph(
    df[icr],
    df["traditional_fitted_predicted_icr"],
    ["Scheduled CIR (g/U)", "Residual (%)"],
    "Residuals for \n Fitted ACE CIR Equation",
)

df["traditional_constants_predicted_icr"] = df.apply(
    lambda x: traditional_constants_icr_equation(x[tdd]), axis=1
)
make_graph(
    df[icr],
    df["traditional_constants_predicted_icr"],
    ["Scheduled CIR (g/U)", "Residual (%)"],
    "Residuals for \n Traditional ACE CIR Equation",
)
