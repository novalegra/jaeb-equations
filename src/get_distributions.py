import pandas as pd
import utils

file_name = "processed-30-min-win_all_selected_rows_2021_02_04_22-v0_1-4d1a82f"
data_path = utils.find_full_path(file_name, ".csv")
df = pd.read_csv(data_path)

distribution_stats = ["mean", "std", "min", "25%", "75%", "max"]
output_df = pd.DataFrame(columns=distribution_stats)
analysis_name = "analyze-distributions"

# Keys for working with exports
suspend_threshold = "suspend_threshold"
correction_lower = "correction_target_lower"
correction_upper = "correction_target_upper"
max_basal = "maximum_basal_rate"
max_bolus = "maximum_bolus"

keys = [suspend_threshold, correction_lower, correction_upper, max_basal, max_bolus]

""" Print out distributions for all the different columns """
for key in keys:
    row = []
    distribution = df[key].describe(include="all")
    for stat in distribution_stats:
        row.append(distribution.loc[stat])

    if len(row) == len(distribution_stats):
        output_df.loc[key] = row

output_df.to_csv(
    utils.get_save_path_with_file(
        file_name,
        analysis_name,
        "distributions_" + utils.get_file_stamps()[0] + ".csv",
        "data-processing",
    )
)
