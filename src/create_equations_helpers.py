import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import utils
from scipy import optimize

LOG_CONSTANT = 1


def calc_exp(arr):

    # remember the original shape
    original_shape = arr.shape

    # flatten the array
    arr_flattened = arr.flatten()

    # initialize an array to store the results
    exp_arr_flattened = np.zeros_like(arr_flattened)

    # apply np.exp() in a loop
    for i in range(len(arr_flattened)):
        exp_arr_flattened[i] = np.exp(arr_flattened[i])

    # reshape back to the original shape
    exp_arr = exp_arr_flattened.reshape(original_shape)

    return exp_arr

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
            "1/TDD": 1 / (all_conditions["TDD"][index]),
            "MIN_OUTPUT": all_conditions["MIN_OUTPUT"][index],
            "MAX_OUTPUT": all_conditions["MAX_OUTPUT"][index],
            "X_intercept": 1,
        }
        output.append(condition_dict)

    return output




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
        try:
            y_estimate = np.exp(y_estimate)
        except:
            y_estimate = calc_exp(y_estimate)

        try:
            y_actual = np.exp(y_actual)
        except:
            y_actual = calc_exp(y_actual)

    residuals = y_estimate - y_actual

    # median absolute percentage error
    absolute_percent_error = np.abs(residuals) / np.maximum(np.abs(y_actual), epsilon)
    loss_score = np.median(absolute_percent_error)

    # load in the right bounding parameters or fitting checks
    if "BASAL" in y_col_name:
        bounding_check_dictionary = make_condition_dicts("basal_fitting_checks")
    elif "ISF" in y_col_name:
        bounding_check_dictionary = make_condition_dicts("isf_fitting_checks")
    elif "CIR" in y_col_name:
        bounding_check_dictionary = make_condition_dicts("cir_fitting_checks")



    for check_dict in bounding_check_dictionary:
        min_val = check_dict["MIN_OUTPUT"]
        max_val = check_dict["MAX_OUTPUT"]

        X_val = [check_dict[param] for param in X_col_names]
        y_pred = equation(fixed_parameters, X_val)
        # print(min_val, max_val, y_pred)
        if "log" in y_col_name:
            warnings.filterwarnings("ignore")
            try:
                y_pred = np.exp(y_pred)
            except:
                y_pred = calc_exp(y_pred)
            warnings.filterwarnings("always")

        if not (min_val <= y_pred <= max_val):
            loss_score = np.inf
            break

    return loss_score



