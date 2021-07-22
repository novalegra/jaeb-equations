import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    median_absolute_error,
    max_error,
    explained_variance_score,
    mean_absolute_percentage_error,
)
from itertools import combinations, combinations_with_replacement, permutations, product
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import statsmodels.api as sm
import utils
from statsmodels.graphics.gofplots import ProbPlot
import pathlib
from tensorflow.keras.losses import Huber
import equation_utils


def adjusted_r_2(r_2, n, k):
    return 1 - ((1 - r_2) * (n - 1) / (n - k - 1))


# Take a list of input parameters and return the list with only
# the parameters that are turned 'on', as per the parameter combination settings
def get_model_inputs(potential_inputs, parameter_settings):
    assert len(potential_inputs) == len(parameter_settings)
    output = []
    for _input, setting in zip(potential_inputs, parameter_settings):
        if setting != "off":
            output.append(_input)

    return output


def no_basal_greater_than_tdd(equation, combo_list):
    model_params = combo_list[2:]
    bmi_setting, carb_setting, tdd_setting = model_params

    not_selected = "off"

    if tdd_setting == not_selected:
        return True

    # Only loop through params that are turned 'on'
    valid_carbs = range(0, 1001, 50) if carb_setting != not_selected else range(1)
    valid_bmis = range(0, 101, 10) if bmi_setting != not_selected else range(1)
    valid_ttds = range(0, 201, 20)

    for carb in valid_carbs:
        for bmi in valid_bmis:
            for tdd in valid_ttds:
                prediction = equation.predict(
                    np.array([get_model_inputs([carb, bmi, tdd], model_params)])
                )
                if prediction > tdd:
                    print(
                        f"Found basal equation with prediction ({prediction}) > TDD ({tdd})"
                    )
                    return False

    return True


# load in data
file_path = utils.find_full_path(
    "2021-05-02_equation_paper_aspirational_data_reduced", ".csv"
)
all_data = pd.read_csv(
    file_path,
    usecols=[
        "geomean_basal_rate",
        "geomean_isf",
        "geomean_weighted_cir",
        "bmi",
        "geomean_total_daily_carbs_in_chunk_outcomes",
        "geomean_total_daily_insulin_dose_in_chunk_outcomes",
    ],
)

all_data.rename(
    columns={
        "geomean_basal_rate": "BR",
        "geomean_isf": "ISF",
        "geomean_weighted_cir": "CIR",
        "bmi": "BMI",
        "geomean_total_daily_carbs_in_chunk_outcomes": "CHO",
        "geomean_total_daily_insulin_dose_in_chunk_outcomes": "TDD",
    },
    inplace=True,
)

# remove any rows where this is a value <= 0
clean_data = all_data[np.sum(all_data <= 0, axis=1) == 0].copy()
clean_data.reset_index(drop=True, inplace=True)

# calculate logs of values
clean_data["BASAL"] = clean_data["BR"] * 24
clean_data["log_BASAL"] = np.log(clean_data["BASAL"])
clean_data["log_BR"] = np.log(clean_data["BR"])
clean_data["log_ISF"] = np.log(clean_data["ISF"])
clean_data["log_CIR"] = np.log(clean_data["CIR"])
clean_data["log_BMI"] = np.log(clean_data["BMI"])
clean_data["log_CHO"] = np.log(clean_data["CHO"])
clean_data["log_TDD"] = np.log(clean_data["TDD"])

y_cols = ["BASAL", "log_BASAL", "ISF", "log_ISF", "CIR", "log_CIR"]

x_cols = [
    "BMI",
    "log_BMI",
    "CHO",
    "log_CHO",
    "TDD",
    "log_TDD",
]

X = clean_data[x_cols]
y = clean_data[y_cols]

# break the data into 70% train and 30% test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1,
)

# add fold information here
kf = KFold(n_splits=5)

# loop through each of the 3 independent variables
for y in [["BASAL", "log_BASAL"], ["ISF", "log_ISF"], ["CIR", "log_CIR"]]:
    print("solving for the {} equations".format(y[0]))

    # consider all combinations
    intercept = [True, False]
    bmi = ["off", "BMI", "log_BMI"]
    cho = ["off", "CHO", "log_CHO"]
    tdd = ["off", "TDD", "log_TDD"]
    all_combos = list(product(y, intercept, bmi, cho, tdd))
    ac_df = pd.DataFrame(all_combos, columns=["y", "intercept", "BMI", "CHO", "TDD"])

    ac_df["n_params"] = np.nan

    for metric in [
        "MAPE",
        "MAE",
        "RMSE",
        "MAX_ERROR",
        "EXPLAIN_VAR",
        "R2",
        "ADJ_R2",
    ]:
        ac_df["{}_test".format(metric)] = np.nan

    for pm in ["intercept", "BMI", "CHO", "TDD"]:
        ac_df["{}_huber".format(pm)] = np.nan
    for pm in ["log_BMI", "log_CHO", "log_TDD"]:
        ac_df["{}_huber".format(pm)] = np.nan

    for combo, ac in enumerate(all_combos):
        print(combo, list(ac))
        [y_lin_log, fit_intercept, bmi_lin_log, cho_lin_log, tdd_lin_log] = ac

        if ("off" in bmi_lin_log) and ("off" in cho_lin_log) and ("off" in tdd_lin_log):
            print("skipping")

        else:
            X_cols = [bmi_lin_log, cho_lin_log, tdd_lin_log]
            X_cols = list(set(X_cols))
            if "off" in X_cols:
                X_cols.remove("off")

            # let's build the full model with all training data so we can test it against our reserved training data
            # fit with huber & grab coefficients/intercept
            huber_regr = linear_model.HuberRegressor(fit_intercept=fit_intercept)
            huber_regr.fit(X_train[X_cols], np.ravel(y_train[y_lin_log]))

            y_predict = huber_regr.predict(X_test[X_cols])

            if fit_intercept:
                ac_df.loc[combo, "intercept_huber"] = huber_regr.intercept_
            for i, key in enumerate(X_train[X_cols].columns):
                ac_df.loc[combo, "{}_huber".format(key)] = huber_regr.coef_[i]

            y_test_vals = y_test[y_lin_log]

            h = Huber(delta=1.35)
            ac_df.loc[combo, "huber_loss_test"] = h(y_test_vals, y_predict).numpy()

            ac_df.loc[combo, "MAPE_test"] = mean_absolute_percentage_error(
                y_test_vals, y_predict
            )
            ac_df.loc[combo, "MAE_test"] = median_absolute_error(y_test_vals, y_predict)
            ac_df.loc[combo, "RMSE_test"] = mean_squared_error(
                y_test_vals, y_predict, squared=False
            )
            ac_df.loc[combo, "MAX_ERROR_test"] = max_error(y_test_vals, y_predict)
            ac_df.loc[combo, "EXPLAIN_VAR_test"] = explained_variance_score(
                y_test_vals, y_predict
            )

            R2 = r2_score(y_test_vals, y_predict)
            ac_df.loc[combo, "R2_test"] = R2
            ac_df.loc[combo, "ADJ_R2_test"] = adjusted_r_2(
                R2, len(y_test_vals), len(X_cols) + (fit_intercept * 1)
            )

            ac_df.loc[combo, "n_params"] = len(X_cols) + (fit_intercept * 0.5)
            ac_df.loc[combo, "pred_greater_than_tdd"] = (
                no_basal_greater_than_tdd(huber_regr, list(ac))
                if "BASAL" in list(ac)[0]
                else False
            )

    print("starting the special equations")

    if "BASAL" in y:
        y_lin_log = "BASAL"
        new_cols = ["rayhan_basal_pred", "trad_basal_pred"]

    if "ISF" in y:
        y_lin_log = "ISF"
        new_cols = ["rayhan_isf_pred", "trad_isf_pred"]

    if "CIR" in y:
        y_lin_log = "CIR"
        new_cols = ["rayhan_icr_pred", "trad_icr_pred"]

    for new_col in new_cols:
        # BASAL EQUATIONS
        if "rayhan_basal_pred" in new_col:
            y_predict = X_test.apply(
                lambda x: equation_utils.jaeb_basal_equation(x["TDD"], x["CHO"]),
                axis=1,
            ).values.reshape(-1, 1)

        if "trad_basal_pred" in new_col:
            y_predict = X_test.apply(
                lambda x: equation_utils.traditional_constants_basal_equation(x["TDD"]),
                axis=1,
            ).values.reshape(-1, 1)

        # ISF EQUATIONS
        if "rayhan_isf_pred" in new_col:
            y_predict = X_test.apply(
                lambda x: equation_utils.jaeb_isf_equation(x["TDD"], x["BMI"]), axis=1,
            ).values.reshape(-1, 1)

        if "trad_isf_pred" in new_col:
            y_predict = X_test.apply(
                lambda x: equation_utils.traditional_constants_isf_equation(x["TDD"]),
                axis=1,
            ).values.reshape(-1, 1)

        # CIR EQUATIONS
        if "rayhan_icr_pred" in new_col:
            y_predict = X_test.apply(
                lambda x: equation_utils.jaeb_icr_equation(x["TDD"], x["CHO"]), axis=1,
            ).values.reshape(-1, 1)

        if "trad_icr_pred" in new_col:
            y_predict = X_test.apply(
                lambda x: equation_utils.traditional_constants_icr_equation(x["TDD"]),
                axis=1,
            ).values.reshape(-1, 1)

        y_test_vals = y_test[y_lin_log]

        ac_df.loc[new_col, "MAPE_test"] = mean_absolute_percentage_error(
            y_test_vals, y_predict
        )

        ac_df.loc[new_col, "MAE_test"] = median_absolute_error(y_test_vals, y_predict)
        ac_df.loc[new_col, "RMSE_test"] = mean_squared_error(
            y_test_vals, y_predict, squared=False
        )
        ac_df.loc[new_col, "MAX_ERROR_test"] = max_error(y_test_vals, y_predict)
        ac_df.loc[new_col, "EXPLAIN_VAR_test"] = explained_variance_score(
            y_test_vals, y_predict
        )

        R2 = r2_score(y_test_vals, y_predict)
        ac_df.loc[new_col, "R2_test"] = R2
        ac_df.loc[new_col, "ADJ_R2_test"] = adjusted_r_2(
            R2, len(y_test_vals), len(X_cols) + (fit_intercept * 1)
        )

    ac_df.sort_values(
        by=["MAPE_test",], ascending=[True], inplace=True,
    )
    ac_df.reset_index(inplace=True)
    ac_df.to_csv("{}-equation-test-results-MAPE-2021-07-21.csv".format(y[0]))
