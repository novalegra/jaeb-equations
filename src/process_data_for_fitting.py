import pandas as pd
from sklearn.model_selection import train_test_split
import utils
import numpy as np

input_file_name = (
    "processed-30-min-win_aggregated_rows_per_patient_2021_01_09_01-v0_1-50aca7a"
)
data_path = utils.find_full_path(input_file_name, ".csv")
df = pd.read_csv(data_path, index_col=0)
analysis_name = "evaluate-equations"
percent_test_data = 0.3


def export(dataframe, df_descriptor, short_file_name="processed-aspirational"):
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


tdd = "avg_total_insulin_per_day_outcomes"
basal = "avg_basal_insulin_per_day_outcomes"  # Total daily basal
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

keys = {
    "age": age,
    "bmi": bmi,
    "bmi_perc": bmi_percentile,
    "total_daily_basal": basal,
    "percent_cgm_available": percent_cgm,
    "days_with_insulin": days_insulin,
    "percent_below_40": percent_below_40,
    "percent_below_54": percent_below_54,
    "percent_70_180": tir,
    "percent_above_250": percent_above_250,
}
print(df.head())
print(df.shape)
aspirational_adults = utils.filter_aspirational_data_adult(df, keys)
aspirational_peds = utils.filter_aspirational_data_peds(df, keys)
aspirational_overall = pd.concat([aspirational_adults, aspirational_peds])
print(
    "{}/{} ({}%) aspirational issue reports, which is {}/{} ({}%) of the patients".format(
        len(aspirational_overall),
        len(df),
        round(len(aspirational_overall) / len(df) * 100),
        len(aspirational_overall[loop_id].unique()),
        len(df[loop_id].unique()),
        round(
            len(aspirational_overall[loop_id].unique())
            / len(df[loop_id].unique())
            * 100
        ),
    )
)
export(aspirational_overall, "overall")

y_cols = [isf, icr, basal, age]

train, test, y_train, y_test = train_test_split(
    aspirational_overall.drop(columns=y_cols).to_numpy(),
    aspirational_overall[y_cols].to_numpy(),
    test_size=percent_test_data,
    random_state=1,
)

combined_train = np.concatenate((train, y_train), axis=1)
combined_test = np.concatenate((test, y_test), axis=1)
column_labels = np.append(
    aspirational_overall.drop(columns=y_cols).columns.values, y_cols,
)
print("Train: {} reports".format(len(combined_train)))
print("Test: {} reports".format(len(combined_test)))

# Reserve a stratified 'percent_test_data' of data for final testing
export(utils.numpy_to_pandas(column_labels, combined_train), "train")
export(utils.numpy_to_pandas(column_labels, combined_test), "test")

utils.find_and_export_kfolds(
    utils.numpy_to_pandas(column_labels, combined_train),
    input_file_name,
    analysis_name,
    utils.DemographicSelection.OVERALL,
    n_splits=5,
)
