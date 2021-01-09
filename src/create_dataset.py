import utils
import pandas as pd
import glob
import os

"""
Create a dataset from 30-minute data intervals

get directory
create output dataset

for each file in directory that matches regex:
    pull the row with the max value of 'cgm_available'

add info from other datasets to cleaned issue report row data
export all rows
"""

# From the individual issue report data
tdd = "avg_total_insulin_per_day_outcomes"
total_daily_basal = "avg_basal_insulin_per_day_outcomes"  # Total daily basal
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
issue_report_date = "issue_report_date"
loop_id = "loop_id"
percent_true = "percent_true_over_outcome"

# From the survey data
bmi = "BMI"
bmi_percentile = "BMIPercentile"
age = "Age"

rows_without_demographic_data = [
    loop_id,
    percent_true,
    percent_cgm,
    tdd,
    total_daily_basal,
    total_daily_carbs,
    isf,
    icr,
    percent_below_40,
    percent_below_54,
    percent_below_70,
    tir,
    percent_above_180,
    percent_above_250,
    percent_above_400,
]

aggregate_output_rows = rows_without_demographic_data.copy()
aggregate_output_rows.extend([age, bmi, bmi_percentile, issue_report_date])

analysis_name = "make_dataset"
all_patient_files = glob.glob(
    os.path.join("..", "jaeb-analysis", "data", ".PHI", "*LOOP*",)
)

all_output_rows_df = None
num_skipped_lack_cgm = 0

for file_path in all_patient_files:
    print("Loading file at {}".format(file_path))
    df = pd.read_csv(file_path)

    # Initialize our df using the column data from our first file
    if all_output_rows_df is None:
        all_output_rows_df = pd.DataFrame(columns=df.columns)

    df.dropna(subset=rows_without_demographic_data)

    # Select the row that highest chance Loop was running for a long time
    best_rows = df[(df[percent_cgm] == df[percent_cgm].max()) & (df[percent_cgm] >= 90)]

    if len(best_rows.index) < 1:
        print(
            "Skipping file at {} due to no rows with >= 90% CGM data".format(file_path)
        )
        num_skipped_lack_cgm += 1
        continue

    all_output_rows_df = all_output_rows_df.append(best_rows.iloc[0], ignore_index=True)

short_file_name = "processed-30-min-win"


def export(dataframe, df_descriptor):
    dataframe.to_csv(
        utils.get_save_path_with_file(
            short_file_name,
            analysis_name,
            short_file_name
            + "_"
            + df_descriptor
            + "_"
            + utils.get_file_stamps()[0]
            + ".csv",
            "dataset-creation",
        )
    )


annotated_issue_reports_df = pd.DataFrame(columns=aggregate_output_rows)
print("We have {} unique patient IDs".format(len(all_output_rows_df[loop_id].unique())))

survey_data_file_name = "Primary-Outcome-Listings"
survey_path = utils.find_full_path(survey_data_file_name, ".csv")
survey_df = pd.read_csv(survey_path)
survey_data_loop_id = "PtID"
num_without_demographics = 0

# Add survey data
for i in range(len(all_output_rows_df.index)):
    selected_row = all_output_rows_df.loc[i]

    assert (
        99.9
        <= selected_row[percent_below_70]
        + selected_row[tir]
        + selected_row[percent_above_180]
        <= 100.1
    )

    patient_id = all_output_rows_df.loc[i, loop_id]
    demographics_rows = survey_df[survey_df[survey_data_loop_id] == patient_id]

    if len(demographics_rows.index) < 1:
        print("Couldn't find demographic info for patient {}".format(patient_id))
        num_without_demographics += 1
        continue

    demographic_row = demographics_rows.iloc[0]
    demographic_bmi = demographic_row[bmi] if demographic_row[bmi] != "." else None
    demographic_bmi_percent = (
        demographic_row[bmi_percentile]
        if demographic_row[bmi_percentile] != "."
        else None
    )

    simplified_row = [
        selected_row[loop_id],
        selected_row[percent_true],
        selected_row[percent_cgm],
        selected_row[tdd],
        selected_row[total_daily_basal],
        selected_row[total_daily_carbs],
        selected_row[isf],
        selected_row[icr],
        selected_row[percent_below_40],
        selected_row[percent_below_54],
        selected_row[percent_below_70],
        selected_row[tir],
        selected_row[percent_above_180],
        selected_row[percent_above_250],
        selected_row[percent_above_400],
        demographic_row[age],
        demographic_bmi,
        demographic_bmi_percent,
        selected_row[issue_report_date],
    ]

    annotated_issue_reports_df.loc[
        len(annotated_issue_reports_df.index)
    ] = simplified_row

num_files = len(all_patient_files)
print(
    "Skipped {}/{} files due to lack of CGM data".format(
        num_skipped_lack_cgm, num_files
    )
)
print(
    "Skipped {}/{} files due to no demographic data".format(
        num_without_demographics, num_files
    )
)
print(annotated_issue_reports_df.head())

export(all_output_rows_df, "all_selected_rows")
export(annotated_issue_reports_df, "aggregated_rows_per_patient")
