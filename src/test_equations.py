import pandas as pd
import utils
import equation_utils

from pumpsettings import PumpSettings
from sklearn.metrics import median_absolute_error, r2_score, mean_squared_error


def compute_statistics(y_true, y_predicted, k):
    # Returns MAE, R^2, and RMSE values
    mae = round(median_absolute_error(y_true, y_predicted), 2)
    r_squared = round(r2_score(y_true, y_predicted), 3)
    adjusted_r_2 = round(utils.adjusted_r_2(r_squared, len(y_predicted), k), 3)
    rmse = round(mean_squared_error(y_true, y_predicted) ** 0.5, 2)
    return (mae, r_squared, adjusted_r_2, rmse)


def run_equation_testing(
    input_file_name,
    jaeb_equations,
    traditional_fitted_equations,
    traditional_constant_equations,
):
    """
    Run equation testing, using the equations from an equation dict
    input_file_name: name of file, without extension
    jaeb_equations: PumpSettings object with equation data
    traditional_equations: PumpSettings object with equation data
    """

    data_path = utils.find_full_path(input_file_name, ".csv")
    df = pd.read_csv(data_path)
    result_cols = [
        "jaeb_mae",
        "jaeb_r_2",
        "jaeb_adj_r_2",
        "jaeb_rmse",
        "traditional_fitted_mae",
        "traditional_fitted_r_2",
        "traditional_fitted_adj_r_2",
        "traditional_fitted_rmse",
        "traditional_constants_mae",
        "traditional_constants_r_2",
        "traditional_constants_adj_r_2",
        "traditional_constants_rmse",
    ]
    output_df = pd.DataFrame(columns=result_cols)
    analysis_name = "evaluate-equations"

    # Keys for working with Jaeb exports
    tdd_key = "avg_total_insulin_per_day_outcomes"
    basal_key = "total_daily_scheduled_basal"  # Total daily basal
    carb_key = "avg_carbs_per_day_outcomes"  # Total daily CHO
    bmi_key = "BMI"
    bmi_percentile = "BMIPercentile"
    isf_key = "avg_isf"
    icr_key = "weighted_cir_outcomes"
    tir_key = "percent_70_180"
    age_key = "Age"

    """ Basal Analysis """
    df["jaeb_predicted_basals"] = df.apply(
        lambda x: jaeb_equations.basal_equation(x[tdd_key], x[carb_key]), axis=1
    )
    df["traditional_fitted_predicted_basals"] = df.apply(
        lambda x: traditional_fitted_equations.basal_equation(x[tdd_key]), axis=1
    )
    df["traditional_constants_predicted_basals"] = df.apply(
        lambda x: traditional_constant_equations.basal_equation(x[tdd_key]), axis=1
    )

    jaeb_basal_stats = compute_statistics(df[basal_key], df["jaeb_predicted_basals"], 2)
    traditional_fitted_basal_stats = compute_statistics(
        df[basal_key], df["traditional_fitted_predicted_basals"], 1
    )
    traditional_constants_basal_stats = compute_statistics(
        df[basal_key], df["traditional_constants_predicted_basals"], 1
    )

    output_df.loc["Basal"] = [
        *jaeb_basal_stats,
        *traditional_fitted_basal_stats,
        *traditional_constants_basal_stats,
    ]

    """ ISF Analysis """
    df["jaeb_predicted_isf"] = df.apply(
        lambda x: jaeb_equations.isf_equation(x[tdd_key], x[bmi_key]), axis=1
    )
    df["traditional_fitted_predicted_isf"] = df.apply(
        lambda x: traditional_fitted_equations.isf_equation(x[tdd_key]), axis=1
    )
    df["traditional_constants_predicted_isf"] = df.apply(
        lambda x: traditional_constant_equations.isf_equation(x[tdd_key]), axis=1
    )
    df = df.dropna(
        subset=[
            "jaeb_predicted_isf",
            "traditional_fitted_predicted_isf",
            "traditional_constants_predicted_isf",
        ]
    )

    jaeb_isf_stats = compute_statistics(df[isf_key], df["jaeb_predicted_isf"], 2)
    traditional_fitted_isf_stats = compute_statistics(
        df[isf_key], df["traditional_fitted_predicted_isf"], 1
    )
    traditional_constants_isf_stats = compute_statistics(
        df[isf_key], df["traditional_constants_predicted_isf"], 1
    )

    output_df.loc["ISF"] = [
        *jaeb_isf_stats,
        *traditional_fitted_isf_stats,
        *traditional_constants_isf_stats,
    ]

    """ ICR Analysis """
    df["jaeb_predicted_icr"] = df.apply(
        lambda x: jaeb_equations.icr_equation(x[tdd_key], x[carb_key]), axis=1
    )
    df["traditional_fitted_predicted_icr"] = df.apply(
        lambda x: traditional_fitted_equations.icr_equation(x[tdd_key]), axis=1
    )
    df["traditional_constants_predicted_icr"] = df.apply(
        lambda x: traditional_constant_equations.icr_equation(x[tdd_key]), axis=1
    )

    jaeb_icr_stats = compute_statistics(df[icr_key], df["jaeb_predicted_icr"], 2)
    traditional_fitted_icr_stats = compute_statistics(
        df[icr_key], df["traditional_fitted_predicted_icr"], 1
    )
    traditional_constants_icr_stats = compute_statistics(
        df[icr_key], df["traditional_constants_predicted_icr"], 1
    )

    output_df.loc["ICR"] = [
        *jaeb_icr_stats,
        *traditional_fitted_icr_stats,
        *traditional_constants_icr_stats,
    ]

    output_df["jaeb_vs_fitted_mae_dif"] = round(
        output_df["jaeb_mae"] - output_df["traditional_fitted_mae"], 2
    )
    output_df["jaeb_vs_fitted_r_2_dif"] = round(
        output_df["jaeb_r_2"] - output_df["traditional_fitted_r_2"], 2
    )
    output_df["jaeb_vs_fitted_rmse_dif"] = round(
        output_df["jaeb_rmse"] - output_df["traditional_fitted_rmse"], 2
    )
    output_df["fitted_vs_constants_mae_dif"] = round(
        output_df["traditional_fitted_mae"] - output_df["traditional_constants_mae"], 2
    )
    output_df["fitted_vs_constants_r_2_dif"] = round(
        output_df["traditional_fitted_r_2"] - output_df["traditional_constants_r_2"], 2
    )
    output_df["fitted_vs_constants_rmse_dif"] = round(
        output_df["traditional_fitted_rmse"] - output_df["traditional_constants_rmse"],
        2,
    )

    short_file_name = (
        input_file_name[0:10] if len(input_file_name) > 10 else input_file_name
    )
    output_df.to_csv(
        utils.get_save_path_with_file(
            input_file_name,
            analysis_name,
            short_file_name + "_equation_errors_" + utils.get_file_stamps()[0] + ".csv",
            "data-analysis",
        )
    )

    df.to_csv(
        utils.get_save_path_with_file(
            input_file_name,
            analysis_name,
            short_file_name
            + "_with_equation_predictions_"
            + utils.get_file_stamps()[0]
            + ".csv",
            "data-analysis",
        )
    )

