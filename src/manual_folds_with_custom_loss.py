import warnings
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import parameter_graphs
import utils
from scipy import optimize
from sklearn.model_selection import KFold, train_test_split

# %% SETTINGS FOR THE USER TO SET
# NOTE: WITH THE WAY THE CODE IS CURRENTLY STRUCTURED IT IS RECOMMENDED THAT YOU RUN EACH Y-VARIABLE
#   & EACH TDD OPTION SEPARATELY. THIS WAS DONE TO ALLOW THE CODE TO BE RUN IN PARALLEL, ALBEIT MANUALLY.
#   IF YOU WANT TO RUN ALL Y_VARIABLES AT ONCE, JUST REPLACE CURRENT LIST WITH THE COMMENTED LIST IN THE LINE BELOW.
Y_VARIABLE_LIST = ["BASAL"]  # ['BASAL', 'log_BASAL', 'ISF', 'log_ISF', 'CIR', 'log_CIR']
# NOTE: IF YOU WANT TO RUN ALL TDD OPTION, COMMENT OUT THE TDD_OPTION HERE
# AND FIND THIS LINE BELOW "tdd = [["off", "TDD", "log_TDD", "1/TDD"][TDD_OPTION]]"
# AND REPLACE IT WITH "tdd = ["off", "TDD", "log_TDD", "1/TDD"]"
# tdd = [["off", "TDD", "log_TDD", "1/TDD"][TDD_OPTION]]  # tdd = ["off", "TDD", "log_TDD", "1/TDD"]
# SELECT A TDD OPTION [0 = "off", 1 = "TDD", 2 = "log_TDD", 3 = "1/TDD"]
TDD_OPTION = 0

MAKE_GRAPHS = False
WORKERS = -1  # set to 1 for debug mode and -1 to use all workers on your machine
VERBOSE = False
LOCAL_SEARCH_ON_TOP_N_RESULTS = 100
LAST_STEP_INTERVAL = 10
SKIP_ALREADY_RUN = False

# %% CONSTANTS
# Small constant to ensure log is never zero
LOG_CONSTANT = 1


def make_condition_dicts(file_name):
    file_path = utils.find_full_path(file_name, ".csv")
    all_conditions = pd.read_csv(file_path)
    output = []

    for index in all_conditions.index:
        condition_dict = {
            "BMI": all_conditions["BMI"][index],
            "log_BMI": np.log(all_conditions["BMI"][index] + LOG_CONSTANT),
            "CHO": all_conditions["CHO"][index],
            "log_CHO": np.log(all_conditions["CHO"][index] + LOG_CONSTANT),
            "TDD": all_conditions["TDD"][index],
            "log_TDD": np.log(all_conditions["TDD"][index] + LOG_CONSTANT),
            "1/TDD": 1 / (all_conditions["TDD"][index] + LOG_CONSTANT),
            "MIN_OUTPUT": all_conditions["MIN_OUTPUT"][index],
            "MAX_OUTPUT": all_conditions["MAX_OUTPUT"][index],
            "X_intercept": 1,
        }
        output.append(condition_dict)

    return output


def get_output_file_name(chunk_index, analysis_type):
    now = datetime.now().strftime("%m-%d-%y")
    return get_output_file_search_name(chunk_index, analysis_type) + f"-{now}.csv"


def get_output_file_search_name(chunk_index, analysis_type):
    return f"{analysis_type}-{TDD_OPTION}-{LOCAL_SEARCH_ON_TOP_N_RESULTS}-equation-results-MAPE-lastindex-{chunk_index}"


basal_check_dicts = make_condition_dicts("basal_fitting_checks")


def brute_optimize(
    X_df,
    y_df,
    objective_function,
    parameter_search_range_tuple,
    equation_function,
    loss_function,
    find_local_min_function=None,
    verbose=False,
    workers=-1,
):
    """
    Brute search optimization with custom equation and loss function

    This is a wrapper or helper function to
    `scipy brute <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brute.html>`_
    function given that the brute function is a little esoteric.

    Parameters
    ----------
    X_df: dataframe
        These are the known X values.
        Here a df is used so that the column names are specified.
        Also, be sure to add the intercept term if needed.
    y_df: dataframe
        These are the known or actual y values.
        Here a df is used so that the column names are specified.
    objective_function : function
        The function you want to minimize.
    parameter_search_range_tuple : tuple
        A tuple that contains slices for each of the parameters_to_estimate
        The search range for each of the parameters using the slice(min, max, step).
        Each parameter should have its own slide -> (slice(min, max, step), slice(min, max, step), etc.)
        CAUTION: increasing the step size or the number of parameters can be computationally expensive
    equation_function : function
        The equation you are trying to fit.
    loss_function : function
        A function with the first two argumets as (y_actual, y_predict) for compatibility with sklearn
    find_local_min_function : function
        Default to None, optimize.fmin is another option


    Returns
    -------
    optimal_parameter_df : dataframe
    meta_dict: dictionary

    # TODO fill in an example

    """
    y_col_name = y_df.columns[0]
    X_col_names = list(X_df.columns)
    X_ndarray = X_df.values

    optimize_args_tuple = (
        equation_function,
        loss_function,
        X_ndarray,
        y_df.values,
        verbose,
        X_col_names,
        y_col_name,
    )

    brute_results = optimize.brute(
        objective_function,
        parameter_search_range_tuple,
        args=optimize_args_tuple,
        full_output=True,
        finish=find_local_min_function,
        disp=verbose,
        workers=workers,
    )

    optimal_parameter_values = brute_results[0]
    optimal_parameter_df = pd.DataFrame(optimal_parameter_values.reshape([1, -1]), columns=X_col_names)

    loss_of_optimal_params = brute_results[1]
    optimal_parameter_df["loss"] = loss_of_optimal_params

    optimal_parameter_df["y_col_name"] = y_col_name
    output_col_order = ["y_col_name"] + ["loss"] + X_col_names
    optimal_parameter_df = optimal_parameter_df[output_col_order]

    search_mesh = brute_results[2]
    search_mesh_loss_scores = brute_results[3]
    search_results_df = pd.DataFrame(search_mesh_loss_scores.reshape([-1, 1]), columns=["loss"])

    fit_equation_string = "{} = ".format(y_col_name)
    for col_idx, X_col_name in enumerate(X_col_names):
        if len(X_col_names) == 1:
            search_results_df[X_col_name] = search_mesh
            if col_idx == len(X_col_names) - 1:
                fit_equation_string += "{} {}".format(round(optimal_parameter_values, 5), X_col_name)
            else:
                fit_equation_string += "{} {} + ".format(round(optimal_parameter_values, 5), X_col_name)
        else:
            search_results_df[X_col_name] = search_mesh[col_idx].reshape([-1, 1])
            if col_idx == len(X_col_names) - 1:
                fit_equation_string += "{} {}".format(round(optimal_parameter_values[col_idx], 5), X_col_name)
            else:
                fit_equation_string += "{} {} + ".format(round(optimal_parameter_values[col_idx], 5), X_col_name)
    if verbose:
        print(fit_equation_string)

    meta_dict = dict()
    meta_dict["fit_equation"] = fit_equation_string
    meta_dict["optimal_parameter_values"] = optimal_parameter_values
    meta_dict["loss_of_optimal_params"] = loss_of_optimal_params
    search_results_df = search_results_df.sort_values(by="loss")
    meta_dict["search_results_df"] = search_results_df.round(5)
    meta_dict["brute_results"] = brute_results
    meta_dict["optimal_parameter_df"] = optimal_parameter_df
    meta_dict["equation_function_name"] = equation_function.__name__
    meta_dict["y_col_name"] = y_col_name
    meta_dict["X_col_names"] = X_col_names
    meta_dict["parameter_search_range_tuple"] = parameter_search_range_tuple
    meta_dict["loss_function"] = loss_function.__name__
    meta_dict["find_local_min_function"] = find_local_min_function

    return optimal_parameter_df.round(5), meta_dict


def custom_objective_function(parameters_to_estimate_1darray, *args_tuple):
    # TODO: add in function header and some checks to make sure the inputs are correct
    (
        equation_function,
        loss_function,
        fixed_parameters_ndarray,
        y_actual,
        verbose,
        X_col_names,
        y_col_name,
    ) = args_tuple
    y_estimate = equation_function(parameters_to_estimate_1darray, fixed_parameters_ndarray)
    loss_score = loss_function(
        y_actual,
        y_estimate,
        equation_function,
        parameters_to_estimate_1darray,
        X_col_names,
        y_col_name,
    )
    if verbose:
        print(parameters_to_estimate_1darray, loss_score)
    return loss_score


def linear_regression_equation(parameters_to_estimate_1darray, fixed_parameters_ndarray):
    parameters_to_estimate_1darray = np.reshape(parameters_to_estimate_1darray, (-1, 1))
    return np.matmul(fixed_parameters_ndarray, parameters_to_estimate_1darray)


def sum_of_squared_errors_loss_function(y_actual, y_estimate):
    return np.sum((y_estimate - y_actual) ** 2)


def custom_basal_loss_with_inf(
    y_actual,
    y_estimate,
    equation,
    fixed_parameters,
    X_col_names,
    y_col_name,
    transform_loss=False,
):
    epsilon = np.finfo(np.float64).eps

    if ("log" in y_col_name) & (transform_loss):
        y_estimate = np.exp(y_estimate)
        y_actual = np.exp(y_actual)

    residuals = y_estimate - y_actual

    # median absolute percentage error
    absolute_percent_error = np.abs(residuals) / np.maximum(np.abs(y_actual), epsilon)
    loss_score = np.median(absolute_percent_error)

    for check_dict in bounding_check_dictionary:
        min_val = check_dict["MIN_OUTPUT"]
        max_val = check_dict["MAX_OUTPUT"]

        X_val = [check_dict[param] for param in X_col_names]
        y_pred = equation(fixed_parameters, X_val)

        if "log" in y_col_name:
            warnings.filterwarnings("ignore")
            y_pred = np.exp(y_pred)
            warnings.filterwarnings("always")

        if not (min_val <= y_pred <= max_val):
            loss_score = np.inf
            break

    return loss_score


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
    for i, m in enumerate([10000, 1000, 100, 10, 1]):
        step = m / 10
        if i <= 2:
            parameter_search_range_tuple = tuple(
                [slice(np.round(-m, 5), np.round(m + step, 5), np.round(step, 5))] * len(X_df.columns)
            )
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

    # load in the right bounding parameters or fitting checks
    if "BASAL" in y_name:
        bounding_check_dictionary = make_condition_dicts("basal_fitting_checks")
    elif "ISF" in y_name:
        bounding_check_dictionary = make_condition_dicts("isf_fitting_checks")
    elif "CIR" in y_name:
        bounding_check_dictionary = make_condition_dicts("cir_fitting_checks")

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

        if MAKE_GRAPHS:
            fixed_parameters = top_result.loc[0, X_cols].values
            parameter_graphs.make_graphs(linear_regression_equation, fixed_parameters, X_cols, y_lin_log)

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

            ac_df.to_csv(get_output_file_name(combo, y_name))

ac_df.reset_index(inplace=True)
ac_df.to_csv(get_output_file_name("final", y_name))
