import pandas as pd
import numpy as np
import utils

input_file_name = (
    "processed-30-min-win_aggregated_rows_per_patient_2021_02_14_23-v0_1-b8c3222"
)
data_path = utils.find_full_path(input_file_name, ".csv")
should_save_graph = True
should_take_log = True

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
bmi = "BMI"
loop_id = "loop_id"


def make_graph(values, axes_labels, title, use_default_labels_if_ln=False):
    if should_save_graph:
        export_path = "Histogram " + title + ".png"
    else:
        export_path = ""

    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if should_take_log:
        values = np.log(abs(values))
        export_path = "Ln " + export_path
        title = "Ln " + title

    plot_limits = [1, 6] if should_take_log and not use_default_labels_if_ln else None
    bin_width = 0.5 if should_take_log and not use_default_labels_if_ln else None

    utils.plot_by_frequency(
        values,
        title,
        axes_labels[1],
        plot_limits,
        bin_width=bin_width,
        export_path=export_path,
        y_min_val=0,
    )


make_graph(
    df[basal],
    ["Count", "Scheduled Basal (U/day)"],
    "Distribution of \nTotal Daily Scheduled Basal",
)

make_graph(
    df[isf], ["Count", "ISF (mg/dL/U)"], "Distribution of ISF",
)

make_graph(
    df[icr], ["Count", "CIR (g/U)"], "Distribution of CIR",
)

make_graph(
    df[tdd], ["Count", "TDD (U)"], "Distribution of TDD",
)

make_graph(
    df[bmi], ["Count", "BMI"], "Distribution of BMI", use_default_labels_if_ln=True
)

make_graph(
    df[total_daily_carbs],
    ["Count", "Daily CHO (g)"],
    "Distribution of Daily CHO Intake",
)
