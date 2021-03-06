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


def rmse(y, y_predict):
    """ RMSE function, as Python doesn't have a library func for it """
    return ((y - y_predict) ** 2).mean() ** 0.5


def adjusted_r_2(r_2, n, k):
    return 1 - ((1 - r_2) * (n - 1) / (n - k - 1))


def extract_bmi_percentile(s):
    """
    Extract a bmi percentile from a string.
    Precondition: string must be in format 'number%' (ex: '20%')
    """
    return int(s[:-1])


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


def three_dimension_plot(x, y, z, labels=["", "", ""], title=""):
    """
    Function to plot a 3D graph of data, with optional labels & a title
    """
    assert len(labels) == 3

    fig = plt.figure(1, figsize=(7, 7))
    ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
    ax.scatter(x, y, z, edgecolor="k", s=50)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])

    plt.title(title, fontsize=25)
    plt.show()
    plt.clf()


def plot_by_frequency(
    column,
    title="",
    x_axis_label="",
    x_lim=None,
    y_min_val=None,
    bins=10,
    export_path="",
    bin_width=None,
    add_density=True,
):
    """
    column - column to plot frequencies of
    title - title of plot
    x_axis_label - label of x axis
    x_lim - list of x limits in form [left, right]
    bins - number of bins to group data into  [select 1 of bins or bin_width]
    should_export - set true to save the plot to png without plotting it
    bin_width - number of entries in each bin, x_lim must be specified [select 1 of bins or bin_width]
    """
    if bin_width and len(x_lim) == 2:
        bins = round((max(column) - min(column)) / bin_width)
    _, d, _ = plt.hist(column, bins=bins, density=add_density)
    if add_density:
        plt.ylabel("Density")
        density = gaussian_kde(column)
        # To get smoother line
        x_smoothed = np.linspace(min(d), max(d), 100)
        spline = make_interp_spline(d, density(d))
        y_smoothed = spline(x_smoothed)

        plt.plot(x_smoothed, y_smoothed, "r--")
    else:
        plt.ylabel("Count of Occurrences")

    plt.title(title, fontsize=25)
    plt.xlabel(x_axis_label)

    if x_lim:
        plt.xlim(x_lim[0], x_lim[1])
    if y_min_val:
        plt.ylim(bottom=y_min_val)
    if len(export_path) > 0:
        plt.savefig(export_path)
    else:
        plt.show()
    plt.clf()


def two_dimension_plot(
    x, y, labels=["", ""], title="", ylim=None, equation=None, legend=None, save=False
):
    """
    Function to plot a 2D graph of data, with optional labels & a title

    ylim: 2 element list, first element is minimum y value and second is max y value
    equation: 2 element list, first element is m and second element is b (y = mx+b form)
    """
    assert len(labels) == 2

    fig = plt.figure(1, figsize=(7, 7))
    plt.scatter(x, y, edgecolor="k", s=50)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    axes = plt.gca()
    if ylim:
        axes.set_ylim(ylim)
    if equation != None:
        x_vals = np.linspace(min(x), max(x), int(max(x) - min(x)))
        m, b = equation[0], equation[1]
        plt.plot(x_vals, x_vals * m + b, "-r")
    if legend != None:
        plt.legend(legend)

    plt.title(title, fontsize=25)
    if save:
        plt.savefig(title + ".png")
    else:
        plt.show()
    plt.clf()


def log_likelihood(n, k, sum_squared_errors, std_dev):
    """
    Find the maximum log likelihood for a *normal* distribution
    Note: formula is from
    https://www.statlect.com/fundamentals-of-statistics/
    normal-distribution-maximum-likelihood
    """
    ll = (
        -(n / 2) * np.log(2 * np.pi)
        - (n / 2) * np.log(std_dev ** 2)
        - 1 / (2 * (std_dev ** 2)) * sum_squared_errors
    )

    return ll


def aic_bic(n, k, sum_squared_errors, std_dev):
    """
    Compute Akaike Information Criterion (AIC) &
    Bayesian Information Criterion (BIC)
    """
    max_log_likelihood = log_likelihood(n, k, sum_squared_errors, std_dev)
    aic = (2 * k) - 2 * max_log_likelihood
    bic = np.log(n) * k - 2 * max_log_likelihood
    return (aic, bic)


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


def get_demographic_export_path(demographic, dataset, analysis_name):
    """ Get file path for export of filtered demographic datasets """
    assert isinstance(demographic, DemographicSelection)
    file_name = demographic.name.lower() + "_" + dataset + ".csv"
    return get_save_path_with_file(dataset, analysis_name, file_name, "data-processing")


def get_figure_export_path(dataset, plot_title, analysis_name):
    """ Get file path for export of filtered demographic datasets """
    short_dataset = dataset[:20] if len(dataset) >= 20 else dataset
    file_name = plot_title + "_" + short_dataset + ".png"
    return get_save_path_with_file(dataset, analysis_name, file_name, "plots")


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
