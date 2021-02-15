import pandas as pd
import utils

distribution_stats = ["mean", "std", "min", "25%", "50%", "75%", "max"]
analysis_name = "analyze-distributions"

# Keys for non-metabolic settings
suspend_threshold = "suspend_threshold"
correction_lower = "correction_target_lower"
correction_upper = "correction_target_upper"
max_basal = "maximum_basal_rate"
max_bolus = "maximum_bolus"

# Keys for settings
bmi = "BMI"
bmi_percentile = "BMIPercentile"
age = "Age"
tdd = "avg_total_insulin_per_day_outcomes"
basal = "total_daily_scheduled_basal"  # Total daily basal
total_daily_carbs = "avg_carbs_per_day_outcomes"  # Total daily CHO
isf = "avg_isf"
icr = "weighted_cir_outcomes"


file_names = [
    "processed-aspirational_overall_2021_02_15_20-v0_1-9eea51f"
    # "test_1_overall_aspirational_2021_02_04_22-v0_1-4d1a82f",
    # "test_2_overall_aspirational_2021_02_04_22-v0_1-4d1a82f",
    # "test_3_overall_aspirational_2021_02_04_22-v0_1-4d1a82f",
    # "test_4_overall_aspirational_2021_02_04_22-v0_1-4d1a82f",
    # "test_5_overall_aspirational_2021_02_04_22-v0_1-4d1a82f",
    # "train_1_overall_aspirational_2021_02_04_22-v0_1-4d1a82f",
    # "train_2_overall_aspirational_2021_02_04_22-v0_1-4d1a82f",
    # "train_3_overall_aspirational_2021_02_04_22-v0_1-4d1a82f",
    # "train_4_overall_aspirational_2021_02_04_22-v0_1-4d1a82f",
    # "train_5_overall_aspirational_2021_02_04_22-v0_1-4d1a82f",
]


def get_and_export_settings(df, file_name):
    # For settings we're 'fitting'
    # keys = [bmi, bmi_percentile, age, tdd, basal, total_daily_carbs, isf, icr]
    # For non-metabolic settings
    keys = [suspend_threshold, correction_lower, correction_upper, max_basal, max_bolus]

    # Filter out unreasonable settings
    df = df[
        (df[suspend_threshold] > 10)
        & (df[correction_lower] > 10)
        & (df[correction_upper] > 10)
    ]

    output_df = pd.DataFrame(columns=distribution_stats)

    """ Print out distributions for all the different columns """
    for key in keys:
        row = []
        distribution = df[key].describe(include="all")
        for stat in distribution_stats:
            row.append(round(distribution.loc[stat], 2))

        if len(row) == len(distribution_stats):
            output_df.loc[key] = row

    output_df.to_csv(
        utils.get_save_path_with_file(
            file_name,
            analysis_name,
            file_name[0:15] + "_distributions_" + utils.get_file_stamps()[0] + ".csv",
            "data-processing",
        )
    )


for f_name in file_names:
    data_path = utils.find_full_path(f_name, ".csv")
    df = pd.read_csv(data_path)
    get_and_export_settings(df, f_name)
