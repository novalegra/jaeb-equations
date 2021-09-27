from itertools import product

import numpy as np
import pandas as pd
import utils
from scipy import optimize
from sklearn.model_selection import KFold, train_test_split


def make_condition_dicts(file_name):
    file_path = utils.find_full_path(file_name, ".csv")
    all_conditions = pd.read_csv(file_path)
    output = []

    for index in all_conditions.index:
        condition_dict = {
            "BMI": all_conditions["BMI"][index],
            "CHO": all_conditions["CHO"][index],
            "TDD": all_conditions["TDD"][index],
            "MIN_OUTPUT": all_conditions["MIN_OUTPUT"][index],
            "MAX_OUTPUT": all_conditions["MAX_OUTPUT"][index],
            "X_intercept": 1,
        }
        output.append(condition_dict)

    return output


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

    # print("fitting {} = {}".format(y_col_name, X_col_names))

    optimize_args_tuple = (
        equation_function,
        loss_function,
        X_ndarray,
        y_df.values,
        verbose,
    )  # , X_col_names)

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
    optimal_parameter_df = pd.DataFrame(
        optimal_parameter_values.reshape([1, -1]), columns=X_col_names
    )

    loss_of_optimal_params = brute_results[1]
    optimal_parameter_df["loss"] = loss_of_optimal_params

    optimal_parameter_df["y_col_name"] = y_col_name
    output_col_order = ["y_col_name"] + ["loss"] + X_col_names
    optimal_parameter_df = optimal_parameter_df[output_col_order]

    search_mesh = brute_results[2]
    search_mesh_loss_scores = brute_results[3]
    search_results_df = pd.DataFrame(
        search_mesh_loss_scores.reshape([-1, 1]), columns=["loss"]
    )

    # TODO: make sure this won't break if number of parameters is 1
    fit_equation_string = "{} = ".format(y_col_name)
    for col_idx, X_col_name in enumerate(X_col_names):
        if len(X_col_names) == 1:
            search_results_df[X_col_name] = search_mesh
            if col_idx == len(X_col_names) - 1:
                fit_equation_string += "{} {}".format(
                    round(optimal_parameter_values, 5), X_col_name
                )
            else:
                fit_equation_string += "{} {} + ".format(
                    round(optimal_parameter_values, 5), X_col_name
                )
        else:
            search_results_df[X_col_name] = search_mesh[col_idx].reshape([-1, 1])
            if col_idx == len(X_col_names) - 1:
                fit_equation_string += "{} {}".format(
                    round(optimal_parameter_values[col_idx], 5), X_col_name
                )
            else:
                fit_equation_string += "{} {} + ".format(
                    round(optimal_parameter_values[col_idx], 5), X_col_name
                )
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
    ) = args_tuple
    y_estimate = equation_function(
        parameters_to_estimate_1darray, fixed_parameters_ndarray
    )
    loss_score = loss_function(
        y_actual, y_estimate
    )  # , equation_function, parameters_to_estimate_1darray)
    if verbose:
        print(parameters_to_estimate_1darray, loss_score)
    return loss_score


def linear_regression_equation(
    parameters_to_estimate_1darray, fixed_parameters_ndarray
):
    parameters_to_estimate_1darray = np.reshape(parameters_to_estimate_1darray, (-1, 1))
    return np.matmul(fixed_parameters_ndarray, parameters_to_estimate_1darray)


def sum_of_squared_errors_loss_function(y_actual, y_estimate):
    return np.sum((y_estimate - y_actual) ** 2)


def custom_basal_loss_with_inf(y_actual, y_estimate, delta=0.65):
    epsilon = np.finfo(np.float64).eps
    residuals = y_estimate - y_actual

    # median absolute percentage error
    absolute_percent_error = np.abs(residuals) / np.maximum(np.abs(y_actual), epsilon)
    loss_score = np.median(absolute_percent_error)

    # %% old code with huber loss
    # outlier_mask = absolute_percent_error > delta
    # loss = np.ones(np.shape(absolute_percent_error)) * np.nan
    # loss[~outlier_mask] = 0.5 * absolute_percent_error[~outlier_mask] ** 2
    # loss[outlier_mask] = delta * (abs(absolute_percent_error[outlier_mask]) - (0.5 * delta))
    # loss_score = np.sum(loss)

    # %% here is a list of custom penalities
    # penalize the loss if any of the estimates over prediction if y_estimate > y_actual,
    # which implies that basal > TDD given that TDD > y_actual for all cases in our dataset
    n_overestimates = np.sum(residuals > 0)
    if n_overestimates > 0:
        loss_score = np.inf

    # add a penalty if any of the estimates are less than 0
    n_y_too_low = np.sum(y_estimate < 0)
    if n_y_too_low > 0:
        loss_score = np.inf

    # add a penalty if any of the estimates are greater than 35 U/hr
    n_y_too_high = np.sum(y_estimate > 35 * 24)
    if n_y_too_high > 0:
        loss_score = np.inf

    # %% this is where we can add in the 19 checks
    # this will look something like y_temp = equation(add in constants from our table (look at google doc)
    # y_temp needs to between min and max basal

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
    # first do a broad
    for i, m in enumerate([100, 10, 1]):  # 10, 1, 0.1, 0.01, 0.001]):
        step = m / 10
        if i == 0:
            parameter_search_range_tuple = tuple(
                [slice(np.round(-m, 5), np.round(m + step, 5), np.round(step, 5))]
                * len(X_df.columns)
            )
            results_df, results_meta_info_dict = brute_optimize(
                X_df=X_df,
                y_df=y_df,
                objective_function=custom_objective_function,
                parameter_search_range_tuple=parameter_search_range_tuple,
                equation_function=linear_regression_equation,
                loss_function=custom_basal_loss_with_inf,
                find_local_min_function=None,  # None,  #optimize.fmin,
                verbose=verbose,
                workers=workers,
            )
            all_brute_results = pd.concat(
                [all_brute_results, results_meta_info_dict["search_results_df"]]
            )
        else:
            local_search_df = all_brute_results.loc[
                all_brute_results["loss"] != np.inf, :
            ].copy()
            local_search_df.drop_duplicates(inplace=True, ignore_index=True)
            local_search_df.sort_values(by="loss", inplace=True)
            local_search_df.reset_index(drop=True, inplace=True)

            # add in a loop here that goes through the length of local_search_df
            for n_local_searches in range(len(local_search_df)):
                # print("searching with {} resolution, around {}".format(m, local_search_df.loc[n_local_searches:n_local_searches, :]))
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
                    find_local_min_function=None,  # None,  #optimize.fmin,
                    verbose=verbose,
                    workers=workers,
                )
                # print("lowest around this point is {}".format(results_df))
                all_brute_results = pd.concat(
                    [all_brute_results, results_meta_info_dict["search_results_df"]]
                )

    # now do a moderate search around the parameter space
    top_wide_search_df = all_brute_results.loc[
        all_brute_results["loss"] != np.inf, :
    ].copy()
    top_wide_search_df.drop_duplicates(inplace=True)
    top_wide_search_df.sort_values(by="loss", inplace=True)
    top_wide_search_df.reset_index(drop=True, inplace=True)
    # print("that took {} seconds".format(time.time() - start_time))

    # do one last brute force search
    parameter_search_range_list = []
    steps = 32
    for col_name in list(X_df.columns):
        min_val = np.round(top_wide_search_df.loc[:, col_name].min(), 5)
        max_val = np.round(top_wide_search_df.loc[:, col_name].max(), 5)
        step_val = np.round((max_val - min_val) / steps, 5)
        if step_val == 0:
            min_val = min_val - 0.001
            max_val = max_val + 0.001
            step_val = np.round((max_val - min_val) / steps, 5)
        parameter_search_range_list.append(
            slice(min_val, np.round(max_val + step_val, 5), step_val)
        )
    parameter_search_range_tuple = tuple(parameter_search_range_list)
    results_df, results_meta_info_dict = brute_optimize(
        X_df=X_df,
        y_df=y_df,
        objective_function=custom_objective_function,
        parameter_search_range_tuple=parameter_search_range_tuple,
        equation_function=linear_regression_equation,
        loss_function=custom_basal_loss_with_inf,
        find_local_min_function=None,  # None,  #optimize.fmin,
        verbose=verbose,
        workers=workers,
    )
    all_brute_results = pd.concat(
        [all_brute_results, results_meta_info_dict["search_results_df"]]
    )

    all_brute_results.sort_values(by="loss", inplace=True)
    all_brute_results.reset_index(drop=True, inplace=True)
    # valid_results_df = all_brute_results.loc[all_brute_results["loss"] != np.inf, :].copy()
    top_result_df = all_brute_results.loc[0:0, :]

    return top_result_df, all_brute_results


# start of code
workers = -1  # set to 1 for debug mode and -1 to use all workers on your machine


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
clean_data["X_intercept"] = 1
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
    "X_intercept",
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

asdf = 4
# loop through each of the 3 independent variables
for y in [
    ["BASAL"]
]:  # [["BASAL", "log_BASAL"]]:  #, ["ISF", "log_ISF"], ["CIR", "log_CIR"]]:
    print("solving for the {} equations".format(y[0]))

    # consider all combinations
    intercept = ["X_intercept"]
    bmi = ["off", "BMI", "log_BMI"]
    cho = ["off", "CHO", "log_CHO"]
    tdd = ["off", "TDD", "log_TDD"]
    all_combos = list(product(y, intercept, bmi, cho, tdd))
    ac_df = pd.DataFrame(all_combos, columns=["y", "X_intercept", "BMI", "CHO", "TDD"])
    ac_df["val_loss"] = np.nan
    for x_beta in x_cols:
        ac_df["beta_{}".format(x_beta)] = np.nan

    for combo, ac in enumerate(all_combos):
        print(combo, list(ac))
        [y_lin_log, x_intercept, bmi_lin_log, cho_lin_log, tdd_lin_log] = ac

        X_cols = [x_intercept, bmi_lin_log, cho_lin_log, tdd_lin_log]
        X_cols = list(set(X_cols))
        if "off" in X_cols:
            X_cols.remove("off")

        # fit with custom loss function
        X_df = pd.DataFrame(X_train[X_cols])
        y_df = pd.DataFrame(y_train[y_lin_log])
        top_result, all_results = fit_equ_with_custom_loss(
            X_df,
            y_df,
            custom_objective_function,
            linear_regression_equation,
            custom_basal_loss_with_inf,
            verbose=False,
            workers=workers,  # -1
        )

        for result_col in top_result.columns:
            if result_col == "train_loss":
                ac_df.loc[combo, "train_loss"] = top_result[result_col].values
            else:
                ac_df.loc[combo, "beta_{}".format(result_col)] = top_result[
                    result_col
                ].values

        # need to take an equation and run it through the custom loss function
        # need to correct the loss values for the log_basal results
        # double check that the seeds do not change
        # see if there are issues with the searching the log space

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
            top_result, all_results = fit_equ_with_custom_loss(
                X_df_train,
                y_df_train,
                custom_objective_function,
                linear_regression_equation,
                custom_basal_loss_with_inf,
                verbose=False,
                workers=workers,  # -1
            )

            # add in the rayhan and traditional equations here and run on custom loss function

            # run fit model on validation set
            X_df_val = pd.DataFrame(X_val_fold[X_cols])
            y_df_val = pd.DataFrame(y_val_fold[y_lin_log])
            y_predict = linear_regression_equation(
                top_result.loc[0, X_cols].values, X_df_val.values
            )
            if "log" in y_lin_log:
                y_predict = np.exp(y_predict)
            val_loss = custom_basal_loss_with_inf(y_df_val.values, y_predict)
            ac_df.loc[combo, "fold_{}_val_loss".format(fold)] = val_loss
            for result_col in top_result.columns:
                ac_df.loc[
                    combo, "fold_{}_train_{}".format(fold, result_col)
                ] = top_result[result_col].values

            if fold == 5:
                val_fold_loss_list = []
                for f in range(1, fold + 1):
                    val_fold_loss_list.append(
                        ac_df.loc[combo, "fold_{}_val_loss".format(f)]
                    )

                val_fold_loss_array = np.array(val_fold_loss_list)

                avg_val_loss = np.mean(
                    val_fold_loss_array[val_fold_loss_array < np.inf]
                )
                ac_df.loc[combo, "val_loss"] = avg_val_loss

ac_df.reset_index(inplace=True)
ac_df.to_csv("{}-equation-results-MAPE-2021-09-26.csv".format(y[0]))
