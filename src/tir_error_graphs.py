import utils
import pandas as pd
from scipy import stats
import equation_utils

input_file_name = "2021-05-02_equation_paper_non_aspirational_data_reduced"
data_path = utils.find_full_path(input_file_name, ".csv")
all_data = pd.read_csv(data_path)
df = pd.read_csv(data_path)

should_save_graph = True

tdd = "geomean_total_daily_insulin_dose_in_chunk_outcomes"
basal = "geomean_basal_rate"  # Hourly basal rate
df[basal] *= 24  # convert to hourly

total_daily_carbs = "geomean_total_daily_carbs_in_chunk_outcomes"  # Total daily CHO
bmi = "bmi"
isf = "geomean_isf"
icr = "geomean_weighted_cir"
tir = "percent_70_180"

result_cols = [
    "slope",
    "intercept",
    "r_value",
    "p_value",
    "std_error",
]

output_df = pd.DataFrame(columns=result_cols)
analysis_name = "tir-evaluate-equations"


def score(predicted, actual):
    return max(predicted / actual, actual / predicted)


def make_graph(
    tir, actual_setting, predicted_setting, axes_labels, title, y_axes=[0, 4]
):
    y = [
        # score(predicted, actual)
        actual / predicted
        for (actual, predicted) in zip(actual_setting, predicted_setting)
    ]

    # Stats: slope, intercept, r_value, p_value, std_err
    stat = stats.linregress(tir, y)

    utils.two_dimension_plot(
        tir,
        y,
        axes_labels,
        title,
        y_axes,
        save=should_save_graph,
        equation=[stat[0], stat[1]],
        legend=["p=" + str(stat[3])],
    )

    print(stat)
    return stat


def make_compound_graph(tir, basal, isf, icr, axes_labels, title):
    basal_pred, basal_actual = basal
    isf_pred, isf_actual = isf
    icr_pred, icr_actual = icr

    assert len(basal_pred) == len(isf_pred) == len(icr_pred)
    # Take max of 'scores' of basal, isf, and icr
    # y = [
    #     max(
    #         score(basal_pred[i], basal_actual[i]),
    #         score(isf_pred[i], isf_actual[i]),
    #         score(icr_pred[i], icr_actual[i]),
    #     )
    #     for i in range(len(basal_pred))
    # ]

    # Take sum of 'scores' of basal, isf, and icr
    y = [
        score(basal_pred[i], basal_actual[i])
        + score(isf_pred[i], isf_actual[i])
        + score(icr_pred[i], icr_actual[i])
        for i in range(len(basal_pred))
    ]

    # Stats: slope, intercept, r_value, p_value, std_err
    stat = stats.linregress(tir, y)

    utils.two_dimension_plot(
        tir,
        y,
        axes_labels,
        title,
        [0, 10],
        save=should_save_graph,
        equation=[stat[0], stat[1]],
        legend=["p=" + str(stat[3])],
    )

    print(stat)
    return stat


df["jaeb_predicted_basal"] = df.apply(
    lambda x: equation_utils.jaeb_basal_equation(x[tdd], x[total_daily_carbs]), axis=1,
)
df["jaeb_predicted_isf"] = df.apply(
    lambda x: equation_utils.jaeb_isf_equation(x[tdd], x[bmi]), axis=1
)
df["jaeb_predicted_icr"] = df.apply(
    lambda x: equation_utils.jaeb_icr_equation(x[tdd], x[bmi]), axis=1
)

drop_list = [
    "jaeb_predicted_basal",
    "jaeb_predicted_isf",
    "jaeb_predicted_icr",
    tir,
    basal,
    isf,
    icr,
]
df = df.dropna(subset=drop_list)
df = df[(df[drop_list] != 0).all(axis=1)]

make_graph(
    df[tir],
    df[basal],
    df["jaeb_predicted_basal"],
    ["TIR", "Basal Actual-Predicted Setting Ratio"],
    "TIR vs Basal Setting Ratio",
)

make_graph(
    df[tir],
    df[isf],
    df["jaeb_predicted_isf"],
    ["TIR", "ISF Actual-Predicted Setting Ratio"],
    "TIR vs ISF Setting Ratio",
)

make_graph(
    df[tir],
    df[icr],
    df["jaeb_predicted_icr"],
    ["TIR", "CIR Actual-Predicted Setting Ratio"],
    "TIR vs CIR Setting Ratio",
)

# make_compound_graph(
#     df[tir],
#     (list(df[basal]), list(df["jaeb_predicted_basal"])),
#     (list(df[isf]), list(df["jaeb_predicted_isf"])),
#     (list(df[icr]), list(df["jaeb_predicted_icr"])),
#     ["TIR", "Error Ratio Sum"],
#     "TIR vs Error Ratio Sum",
# )
