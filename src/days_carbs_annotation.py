import utils
import pandas as pd
import glob
import os

analysis_name = "make_dataset"
all_patient_files = glob.glob(
    os.path.join("..", "jaeb-analysis", "data", ".PHI", "*LOOP*",)
)

carbs_per_chunk = "n_carbs_in_chunk"
date = "date"
issue_report_date = "issue_report_date"
short_file_name = "carb_annotated"
days_carbs = "days_carb_data"


def export(dataframe, df_descriptor):
    dataframe.to_csv(
        utils.get_save_path_with_file(
            short_file_name,
            analysis_name,
            df_descriptor
            + "_"
            + short_file_name
            + "_"
            + utils.get_file_stamps()[0]
            + ".csv",
            "dataset-creation",
        )
    )


for i, file_path in enumerate(all_patient_files):
    print("Loading file at {} ({}/{})".format(file_path, i + 1, len(all_patient_files)))
    df = pd.read_csv(file_path)

    days_with_carb_data = len(
        set(df[(df[carbs_per_chunk] > 0) & (df[date] <= df[issue_report_date])][date])
    )
    if days_with_carb_data < 14:
        print("! {} days w/carbs".format(days_with_carb_data))
    df[days_carbs] = days_with_carb_data

    file_name = file_path.split("/")[-1].split(".")[0]
    export(df, file_name)

