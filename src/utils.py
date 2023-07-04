import math
import os
import subprocess

import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from sklearn.model_selection import KFold
from enum import Enum
from scipy.stats import gaussian_kde
from scipy.interpolate import make_interp_spline


class DemographicSelection(Enum):
    OVERALL = 1
    PEDIATRIC = 2
    ADULT = 3
    ASPIRATIONAL = 4
    NON_ASPIRATIONAL = 5


def numpy_to_pandas(column_names, data):
    return pd.DataFrame(data=data, columns=column_names)


def filter_aspirational_data_adult(df, keys):
    adults = df[df[keys["age"]] >= 18]
    before_size = len(adults)
    adults = adults[
        # Normal weight
        (df[keys["bmi"]] < 25)
        & (df[keys["bmi"]] >= 18.5)
    ]
    print("Adult BMI filter: {} -> {} pts".format(before_size, len(adults)))
    return filter_aspirational_data_without_weight(adults, keys)


def filter_aspirational_data_peds(df, keys):
    peds = df[(df[keys["age"]] < 18) & (df[keys["bmi_perc"]] != ".")]
    before_size = len(peds)
    peds = peds[
        # Normal weight
        (peds[keys["bmi_perc"]] < 0.85)
        & (peds[keys["bmi_perc"]] >= 0.05)
    ]
    print("Peds BMI filter: {} -> {} pts".format(before_size, len(peds)))
    return filter_aspirational_data_without_weight(peds, keys)


def filter_aspirational_data_without_weight(df, keys):
    if "days_with_insulin" in keys and keys["days_with_insulin"] is not None:
        df = df[df[keys["days_with_insulin"] >= 14]]

    return df[
        (df[keys["total_daily_basal"]] > 1)
        # Enough data to evaluate
        & (df[keys["percent_cgm_available"]] >= 90)
        # Good CGM distributions
        & (df[keys["percent_below_54"]] < 1)
        & (df[keys["percent_below_70"]] < 4)
        & (df[keys["percent_70_180"]] > 70)
        & (df[keys["percent_above_250"]] < 5)
    ]


def find_full_path(resource_name, extension):
    """ Find file path, given name and extension
        example: "/home/pi/Media/tidepool_demo.json"

        This will return the *first* instance of the file

    Arguments:
    resource_name -- name of file without the extension
    extension -- ending of file (ex: ".json")

    Output:
    path to file
    """
    search_dir = Path(__file__).parent.parent
    for root, dirs, files in os.walk(search_dir):
        for name in files:
            (base, ext) = os.path.splitext(name)
            if base == resource_name and extension == ext:
                return os.path.join(root, name)

    raise Exception("No file found for specified resource name & extension")


def file_exists(file_name, extension, use_startswith=False, search_dir=Path(__file__).parent.parent):
    """ Find if a file exists, given name and extension

    Arguments:
    file_name -- name of file without the extension
    extension -- ending of file (ex: ".json")
    use_startswith -- whether to check if the file starts with the name, rather than being an exact match

    Output:
    path to file
    """
    # search_dir = Path(__file__).parent.parent
    for root, dirs, files in os.walk(search_dir):
        for name in files:
            (base, ext) = os.path.splitext(name)
            if use_startswith and base.startswith(file_name) and extension == ext:
                return True
            elif not use_startswith and base == file_name and extension == ext:
                return True

    return False


def find_matching_file_name(key, extension, search_dir):
    """Find file path, given key and extension
        example: "/home/pi/Media/tidepool_demo.json"

        This will return the *first* instance of the file

    Arguments:
    key -- portion of the file name that should match
    extension -- ending of file (ex: ".json")
    search_dir -- directory to search in

    Output:
    file name
    """
    for root, dirs, files in os.walk(search_dir):
        for name in files:

            (base, ext) = os.path.splitext(name)
            if key in base and extension == ext:
                return base

    raise Exception("No file found for specified resource key & extension")


def get_file_stamps():
    """
    Get context for information generated at runtime.
    """
    current_commit = (
        subprocess.check_output(["git", "describe", "--always"]).strip().decode("utf-8")
    )
    utc_string = dt.datetime.utcnow().strftime("%Y_%m_%d_%H")
    code_description = "v0_1"
    date_version_name = "{}-{}-{}".format(utc_string, code_description, current_commit)

    return date_version_name, utc_string, code_description, current_commit


def make_dir_if_it_doesnt_exist(dir_):
    if not os.path.isdir(dir_):
        os.makedirs(dir_)


def get_save_path(
    dataset_name, full_analysis_name, report_type="figures", root_dir=".reports"
):
    output_path = os.path.join(root_dir, dataset_name)
    date_version_name, _, _, _ = get_file_stamps()
    save_path = os.path.join(
        output_path, "{}-{}".format(full_analysis_name, date_version_name), report_type
    )

    make_dir_if_it_doesnt_exist(save_path)

    return save_path


def get_save_path_with_file(
    dataset_name,
    full_analysis_name,
    file_name,
    report_type="figures",
    root_dir=".reports",
):
    return os.path.join(
        get_save_path(dataset_name, full_analysis_name, report_type, root_dir),
        file_name,
    )


def find_and_export_kfolds(df, input_file_name, analysis_name, demographic, n_splits=5):
    assert isinstance(demographic, DemographicSelection)
    # Set random state so results are reproduceable
    kf = KFold(n_splits=n_splits, random_state=2, shuffle=True)

    group = 1
    for train_indexes, test_indexes in kf.split(df):
        df.iloc[train_indexes].reset_index(drop=True).to_csv(
            get_save_path_with_file(
                input_file_name,
                analysis_name,
                "train_"
                + str(group)
                + "_"
                + demographic.name.lower()
                + "_aspirational_"
                + get_file_stamps()[0]
                + ".csv",
                "data-processing",
            )
        )
        df.iloc[test_indexes].reset_index(drop=True).to_csv(
            get_save_path_with_file(
                input_file_name,
                analysis_name,
                "test_"
                + str(group)
                + "_"
                + demographic.name.lower()
                + "_aspirational_"
                + get_file_stamps()[0]
                + ".csv",
                "data-processing",
            )
        )
        group += 1
