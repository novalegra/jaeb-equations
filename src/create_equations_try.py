import os
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import utils
from create_equations_helpers import (
    brute_optimize,
    custom_objective_function,
    linear_regression_equation,
    custom_basal_loss_with_inf,
)
from sklearn.model_selection import KFold, train_test_split

# %% SETTINGS FOR THE USER TO SET
# NOTE: WITH THE WAY THE CODE IS CURRENTLY STRUCTURED IT IS RECOMMENDED THAT YOU RUN EACH Y-VARIABLE
#   & EACH TDD OPTION SEPARATELY, SINCE EACH RUN TAKES SOME TIME. THIS WAS DONE TO ALLOW THE CODE TO BE RUN IN PARALLEL,
#   ALBEIT MANUALLY. IF YOU WANT TO RUN ALL Y_VARIABLES AT ONCE, JUST REPLACE CURRENT LIST WITH THE COMMENTED LIST IN THE LINE BELOW.
Y_VARIABLE_LIST = ["CIR"]  # ['BASAL', 'log_BASAL', 'ISF', 'log_ISF', 'CIR', 'log_CIR']
# NOTE: IF YOU WANT TO RUN ALL TDD OPTION, COMMENT OUT THE TDD_OPTION HERE
# AND FIND THIS LINE BELOW "tdd = [["off", "TDD", "log_TDD", "1/TDD"][TDD_OPTION]]"
# AND REPLACE IT WITH "tdd = ["off", "TDD", "log_TDD", "1/TDD"]"
# tdd = [["off", "TDD", "log_TDD", "1/TDD"][TDD_OPTION]]  # tdd = ["off", "TDD", "log_TDD", "1/TDD"]
# SELECT A TDD OPTION [0 = "off", 1 = "TDD", 2 = "log_TDD", 3 = "1/TDD"]
TDD_OPTION = 3

WORKERS = 1  # set to 1 for debug mode and -1 to use all workers on your machine
VERBOSE = True
LOCAL_SEARCH_ON_TOP_N_RESULTS = 100
LAST_STEP_INTERVAL = 100
SKIP_ALREADY_RUN = True

# %% CONSTANTS
# Small constant to ensure log is never zero
LOG_CONSTANT = 1

if "ISF" in Y_VARIABLE_LIST[0]:
    y_range = 2000
    y_steps = 100
elif "CIR" in Y_VARIABLE_LIST[0]:
    y_range = 500
    y_steps = 50
elif "BASAL" in Y_VARIABLE_LIST[0]:
    y_range = 10
    y_steps = 0.5

def get_output_file_search_name(chunk_index, analysis_type):
    return f"{analysis_type}-{TDD_OPTION}-{LOCAL_SEARCH_ON_TOP_N_RESULTS}-equation-results-MAPE-lastindex-{chunk_index}"


def get_output_file_name(chunk_index, analysis_type):
    now = datetime.now().strftime("%m-%d-%y")
    return get_output_file_search_name(chunk_index, analysis_type) + f"-{now}.csv"


def fit_equ_with_custom_loss(
    X_df,
    y_df,
    custom_objective_function,
    linear_regression_equation,
    custom_basal_loss_with_inf,
    verbose=False,
    workers=-1,
):
    all_brute_results = pd.DataFrame(columns=["loss"] + list(X_df.columns))
    # first do a broad search
    for i, m in enumerate([y_range, 1000, 100, 10, 1]):
        step = m / 10
        print(f"working on grid size of -{m} by {m} at {datetime.now()}")
        if i <= 0:
            step = y_steps
            parameter_search_range_tuple = tuple(
                [slice(np.round(-m, 5), np.round(m + step, 5), np.round(step, 5))] * len(X_df.columns)
            )
            # print(parameter_search_range_tuple)
            results_df, results_meta_info_dict = brute_optimize(
                X_df=X_df,
                y_df=y_df,
                objective_function=custom_objective_function,
                parameter_search_range_tuple=parameter_search_range_tuple,
                equation_function=linear_regression_equation,
                loss_function=custom_basal_loss_with_inf,
                find_local_min_function=None,
                verbose=verbose,
                workers=workers,
            )
            all_brute_results = pd.concat([all_brute_results, results_meta_info_dict["search_results_df"]])
        else:
            local_search_df = all_brute_results.loc[all_brute_results["loss"] != np.inf, :].copy()
            local_search_df.drop_duplicates(inplace=True, ignore_index=True)
            local_search_df.sort_values(by="loss", inplace=True)
            local_search_df.reset_index(drop=True, inplace=True)

            # add in a loop here that goes through the length of local_search_df
            for n_local_searches in range(min(LOCAL_SEARCH_ON_TOP_N_RESULTS, len(local_search_df))):
                parameter_search_range_list = []
                for col_name in list(X_df.columns):
                    local_val = local_search_df.loc[n_local_searches, col_name]
                    parameter_search_range_list.append(
                        slice(
                            np.round(local_val - m, 5),
                            np.round(local_val + m + step, 5),
                            np.round(step, 5),
                        )
                    )
                parameter_search_range_tuple = tuple(parameter_search_range_list)

                results_df, results_meta_info_dict = brute_optimize(
                    X_df=X_df,
                    y_df=y_df,
                    objective_function=custom_objective_function,
                    parameter_search_range_tuple=parameter_search_range_tuple,
                    equation_function=linear_regression_equation,
                    loss_function=custom_basal_loss_with_inf,
                    find_local_min_function=None,
                    verbose=verbose,
                    workers=workers,
                )

                all_brute_results = pd.concat([all_brute_results, results_meta_info_dict["search_results_df"]])

    # now do a moderate search around the parameter space
    top_wide_search_df = all_brute_results.loc[all_brute_results["loss"] != np.inf, :].copy()
    top_wide_search_df.drop_duplicates(inplace=True)
    top_wide_search_df.sort_values(by="loss", inplace=True)
    top_wide_search_df.reset_index(drop=True, inplace=True)

    # If we couldn't find a non-inf loss, we failed to find a fit
    if len(top_wide_search_df) < 1:
        return None, None, False

    # do one last brute force search
    parameter_search_range_list = []
    steps = LAST_STEP_INTERVAL
    for col_name in list(X_df.columns):
        min_val = np.round(top_wide_search_df.loc[:, col_name].min(), 5)
        max_val = np.round(top_wide_search_df.loc[:, col_name].max(), 5)
        step_val = np.round((max_val - min_val) / steps, 5)
        if step_val == 0:
            min_val = min_val - 0.001
            max_val = max_val + 0.001
            step_val = np.round((max_val - min_val) / steps, 5)
        parameter_search_range_list.append(slice(min_val, np.round(max_val + step_val, 5), step_val))
    parameter_search_range_tuple = tuple(parameter_search_range_list)
    results_df, results_meta_info_dict = brute_optimize(
        X_df=X_df,
        y_df=y_df,
        objective_function=custom_objective_function,
        parameter_search_range_tuple=parameter_search_range_tuple,
        equation_function=linear_regression_equation,
        loss_function=custom_basal_loss_with_inf,
        find_local_min_function=None,
        verbose=verbose,
        workers=workers,
    )
    all_brute_results = pd.concat([all_brute_results, results_meta_info_dict["search_results_df"]])

    all_brute_results.sort_values(by="loss", inplace=True)
    all_brute_results.reset_index(drop=True, inplace=True)
    top_result_df = all_brute_results.loc[0:0, :]
    return top_result_df, all_brute_results, True


# %% START OF CODE

# load in data
file_path = utils.find_full_path("2021-05-02_equation_paper_aspirational_data_reduced", ".csv")
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

clean_data["X_intercept"] = 1
clean_data["BASAL"] = clean_data["BR"] * 24
clean_data["log_BASAL"] = np.log(clean_data["BASAL"] + LOG_CONSTANT)
clean_data["log_BR"] = np.log(clean_data["BR"] + LOG_CONSTANT)
clean_data["log_ISF"] = np.log(clean_data["ISF"] + LOG_CONSTANT)
clean_data["log_CIR"] = np.log(clean_data["CIR"] + LOG_CONSTANT)
clean_data["log_BMI"] = np.log(clean_data["BMI"] + LOG_CONSTANT)
clean_data["log_CHO"] = np.log(clean_data["CHO"] + LOG_CONSTANT)
clean_data["log_TDD"] = np.log(clean_data["TDD"] + LOG_CONSTANT)
clean_data["1/TDD"] = 1 / clean_data["TDD"]

y_cols = ["BASAL", "log_BASAL", "ISF", "log_ISF", "CIR", "log_CIR"]

x_cols = ["X_intercept", "BMI", "log_BMI", "CHO", "log_CHO", "TDD", "log_TDD", "1/TDD"]

X = clean_data[x_cols]
y = clean_data[y_cols]

# break the data into 70% train and 30% test set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=1,
)

# loop through each of the 3 independent variables
for y_name in Y_VARIABLE_LIST:
    print(f"solving for the {y_name} equations")


    # consider all combinations
    intercept = ["off", "X_intercept"]
    bmi = ["off", "BMI", "log_BMI"]
    cho = ["off", "CHO", "log_CHO"]
    # NOTE: IF YOU WANT TO RUN ALL TDD OPTIONS, REPLACE THE CURRENT LINE WITH THE COMMENTED LINE, IN THE LINE BELOW
    tdd = [["off", "TDD", "log_TDD", "1/TDD"][TDD_OPTION]]  # tdd = ["off", "TDD", "log_TDD", "1/TDD"]
    all_combos = list(product([y_name], intercept, bmi, cho, tdd))
    ac_df = pd.DataFrame(all_combos, columns=["y", "X_intercept", "BMI", "CHO", "TDD"])
    ac_df["val_loss"] = np.nan
    for x_beta in x_cols:
        ac_df["beta_{}".format(x_beta)] = np.nan

    for combo, ac in enumerate(all_combos):
        if SKIP_ALREADY_RUN:
            if utils.file_exists(get_output_file_search_name(combo, y_name), ".csv", use_startswith=True):
                print(f"Skipping combo {combo} since we have data for it")
                continue

        print(combo, list(ac))
        [y_lin_log, x_intercept, bmi_lin_log, cho_lin_log, tdd_lin_log] = ac

        X_cols = [x_intercept, bmi_lin_log, cho_lin_log, tdd_lin_log]
        X_cols = list(set(X_cols))
        if "off" in X_cols:
            X_cols.remove("off")

        if len(X_cols) == 0:
            print(f"Skipping combo {combo} because all parameters are off")
            continue

        # fit with custom loss function
        X_df = pd.DataFrame(X_train[X_cols])
        y_df = pd.DataFrame(y_train[y_lin_log])
        top_result, all_results, success = fit_equ_with_custom_loss(
            X_df,
            y_df,
            custom_objective_function,
            linear_regression_equation,
            custom_basal_loss_with_inf,
            verbose=VERBOSE,
            workers=WORKERS,
        )

        if not success:
            print(f"ERROR: unable to find fit for {list(ac)} parameters")
            continue

        for result_col in top_result.columns:
            if result_col == "train_loss":
                ac_df.loc[combo, "train_loss"] = top_result[result_col].values
            else:
                ac_df.loc[combo, "beta_{}".format(result_col)] = top_result[result_col].values

        # break the training set into 5 folds for model selection with cross validation
        kf = KFold(n_splits=5)
        fold = 0
        for i, j in kf.split(X_train):
            fold = fold + 1
            print("starting fold {}".format(fold))
            X_train_fold = X_train.iloc[i, :]
            X_val_fold = X_train.iloc[j, :]
            y_train_fold = y_train.iloc[i, :]
            y_val_fold = y_train.iloc[j, :]

            # fit with custom loss function
            X_df_train = pd.DataFrame(X_train_fold[X_cols])
            y_df_train = pd.DataFrame(y_train_fold[y_lin_log])
            top_result, all_results, success = fit_equ_with_custom_loss(
                X_df_train,
                y_df_train,
                custom_objective_function,
                linear_regression_equation,
                custom_basal_loss_with_inf,
                verbose=VERBOSE,
                workers=WORKERS,
            )

            # run fit model on validation set
            X_df_val = pd.DataFrame(X_val_fold[X_cols])
            y_df_val = pd.DataFrame(y_val_fold[y_lin_log])
            fixed_parameters = top_result.loc[0, X_cols].values
            vals = X_df_val.values
            y_predict = linear_regression_equation(fixed_parameters, X_df_val.values)

            fold_X_col_names = list(X_df_val.columns)
            val_loss = custom_basal_loss_with_inf(
                y_df_val.values,
                y_predict,
                linear_regression_equation,
                fixed_parameters,
                fold_X_col_names,
                y_lin_log,
                transform_loss=True,
            )
            ac_df.loc[combo, "fold_{}_val_loss".format(fold)] = val_loss
            for result_col in top_result.columns:
                ac_df.loc[combo, "fold_{}_train_{}".format(fold, result_col)] = top_result[result_col].values

            if fold == 5:
                val_fold_loss_list = []
                for f in range(1, fold + 1):
                    val_fold_loss_list.append(ac_df.loc[combo, "fold_{}_val_loss".format(f)])

                val_fold_loss_array = np.array(val_fold_loss_list)

                avg_val_loss = np.mean(val_fold_loss_array[val_fold_loss_array < np.inf])
                ac_df.loc[combo, "val_loss"] = avg_val_loss

            ac_df.to_csv(os.path.join("results", get_output_file_name(combo, y_name)))

ac_df.reset_index(inplace=True)
ac_df.to_csv(os.path.join("results", get_output_file_name("final", y_name)))
