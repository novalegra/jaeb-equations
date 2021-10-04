from itertools import product
import matplotlib.pyplot as plt
import numpy as np

BMI_PARAMS = [12, 25, 45]
CARB_PARAMS = [0, 250, 500]

# Small constant to ensure log is never zero
LOG_CONSTANT = 1


def get_param_combos():
    return product(BMI_PARAMS, CARB_PARAMS)


def make_graphs(equation, parameters_to_estimate, X_col_names):
    """
    equation: equation to use to generate plots
    parameters_to_estimate: coefficients of parameters in equation
    X_col_names: names of parameters in equation, in order of input
    """
    if not (
        ("BMI" in X_col_names or "log_BMI" in X_col_names)
        and ("CHO" in X_col_names or "log_CHO" in X_col_names)
        and ("TDD" in X_col_names or "log_TDD" in X_col_names)
    ):
        print("ERROR: unable to run because BMI, CHO, or TDD is missing as a parameter")
        return

    fig, axs = plt.subplots(3, 3)

    cache = []
    max_y, min_y = -float("inf"), float("inf")

    for i, combo in enumerate(get_param_combos()):
        x, y = get_x_y(combo, equation, parameters_to_estimate, X_col_names)
        max_y = max(max_y, max(y))
        min_y = min(min_y, min(y))
        cache.append((x, y, i, combo))

    for x, y, i, combo in cache:
        make_graph(
            axs[i // 3][i % 3],
            x,
            y,
            f"BMI {combo[0]}, CHO {combo[1]}",
            ylim=[min_y, max_y],
        )

    fig.suptitle("Basal vs TDD")
    plt.tight_layout()
    plt.show()


def get_x_y(combo, equation, parameters_to_estimate, X_col_names):
    bmi, cho = combo
    param_dict = {
        "BMI": bmi,
        "log_BMI": np.log(bmi + LOG_CONSTANT),
        "CHO": cho,
        "log_CHO": np.log(cho + LOG_CONSTANT),
        "X_intercept": 1,
    }

    tdd_range = range(0, 500, 5)
    y_preds = []

    for tdd in tdd_range:
        param_dict["TDD"] = tdd
        param_dict["log_TDD"] = np.log(tdd + LOG_CONSTANT)

        fixed_parameters = [param_dict[param] for param in X_col_names]

        y_pred = equation(parameters_to_estimate, fixed_parameters)
        if y_pred is None:
            print("CAUTION!! y_pred is none")

        y_preds.append(y_pred)

    return tdd_range, y_preds


def make_graph(ax, x, y, title, ylim=None):
    ax.plot(x, y)
    ax.set_title(title)

    if ylim is not None:
        ax.set_ylim(ylim)


def linear_regression_equation(
    parameters_to_estimate_1darray, fixed_parameters_ndarray
):
    parameters_to_estimate_1darray = np.reshape(parameters_to_estimate_1darray, (-1, 1))
    return np.matmul(fixed_parameters_ndarray, parameters_to_estimate_1darray)


# Example call
# make_graphs(
#     linear_regression_equation, [0.15, 0.5, 0.35], ["log_CHO", "log_TDD", "BMI"]
# )
