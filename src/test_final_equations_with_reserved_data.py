from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
import os
import pandas as pd
import utils
from sklearn.model_selection import train_test_split

# LOAD IN FILES
# file_name = "BASAL-ALL-100-equation-results-MAPE-lastindex-final-06-01-22.csv"
# file_name = "ISF-ALL-100-equation-results-MAPE-lastindex-final-06-01-22.csv"
file_name = "CIR-ALL-100-equation-results-MAPE-lastindex-final-06-01-22.csv"

# CONSTANTS
EPSILON = np.finfo(np.float64).eps
LOG_CONSTANT = 1

# MAKE OUTPUT DIRECTORY FOR PLOTS
plot_directory = file_name.replace(".csv", " PLOTS")
if not os.path.isdir(plot_directory):
    os.makedirs(plot_directory)

def linear_regression_equation(parameters_to_estimate_1darray, fixed_parameters_ndarray):
    parameters_to_estimate_1darray = np.reshape(parameters_to_estimate_1darray, (-1, 1))
    return np.matmul(fixed_parameters_ndarray, parameters_to_estimate_1darray)


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

# %% START OF NEW CODE
keep_columns = [
    "y",
    "X_intercept",
    "BMI",
    "CHO",
    "TDD",
    "beta_X_intercept",
    "beta_BMI",
    "beta_log_BMI",
    "beta_CHO",
    "beta_log_CHO",
    "beta_TDD",
    "beta_log_TDD",
    "beta_1/TDD",
    "val_loss",
]

beta_columns = [
    "beta_X_intercept",
    "beta_BMI",
    "beta_log_BMI",
    "beta_CHO",
    "beta_log_CHO",
    "beta_TDD",
    "beta_log_TDD",
    "beta_1/TDD",
]

x_columns = ["X_intercept", "BMI", "CHO", "TDD"]


df = pd.read_csv(file_name)

# only keep models that have a cross validation model score (val_loss)
converged_model_mask = df["val_loss"].notnull()
# only keep models that have beta_coefficients not equal to 0
non_zero_coefficients_mask = np.sum(df[beta_columns] == 0, axis=1) == 0

final_equation_mask = converged_model_mask & non_zero_coefficients_mask
final_equations_df = df.loc[final_equation_mask, keep_columns].copy().reset_index(drop=True)

if "BASAL" in file_name:
    final_equations_df.loc[
        final_equations_df.index.max() + 1, ["y", "X_intercept", "BMI", "CHO", "TDD", "beta_TDD"]
    ] = ["BASAL", "off", "off", "off", "TDD", 0.5]

elif "ISF" in file_name:
    final_equations_df.loc[
        final_equations_df.index.max() + 1, ["y", "X_intercept", "BMI", "CHO", "TDD", "beta_1/TDD"]
    ] = ["ISF", "off", "off", "off", "1/TDD", 1700]

elif "CIR" in file_name:
    final_equations_df.loc[
        final_equations_df.index.max() + 1, ["y", "X_intercept", "BMI", "CHO", "TDD", "beta_1/TDD"]
    ] = ["CIR", "off", "off", "off", "1/TDD", 450]


final_equations_df["test_mdape"] = np.nan
final_equations_df["test_mape"] = np.nan
final_equations_df["test_rmse"] = np.nan
final_equations_df["equation"] = ""

for i in final_equations_df.index:
    y_lin_log = final_equations_df.loc[i, "y"]
    x_intercept, bmi_lin_log, cho_lin_log, tdd_lin_log = final_equations_df.loc[i, x_columns]
    X_cols = [x_intercept, bmi_lin_log, cho_lin_log, tdd_lin_log]
    X_cols = [x for x in X_cols if "off" not in x]

    # add prefix of beta_ to each column in the equation
    beta_coefficient_columns = ["beta_" + sub for sub in X_cols]
    fixed_parameters = final_equations_df.loc[i, beta_coefficient_columns].values

    equation_string = f"{final_equations_df.loc[i, 'y']} = "
    for j, beta_coefficient_column in enumerate(beta_coefficient_columns):
        if j < len(beta_coefficient_columns) - 1:
            if "intercept" in X_cols[j]:
                equation_string = f"{equation_string} {fixed_parameters[j]} + "
            elif "log" in X_cols[j]:
                equation_string = f"{equation_string} {fixed_parameters[j]} log({X_cols[j].strip('log_')} + 1) + "
            elif "1/" in X_cols[j]:
                equation_string = f"{equation_string} {fixed_parameters[j]} / ({X_cols[j].strip('1/')} + 1) + "
            else:
                equation_string = f"{equation_string} {fixed_parameters[j]} * {X_cols[j]} + "
        else:
            if "intercept" in X_cols[j]:
                equation_string = f"{equation_string} {fixed_parameters[j]}"
            elif "log" in X_cols[j]:
                equation_string = f"{equation_string} {fixed_parameters[j]} log({X_cols[j].strip('log_')} + 1)"
            elif "1/" in X_cols[j]:
                equation_string = f"{equation_string} {fixed_parameters[j]} / ({X_cols[j].strip('1/')} + 1)"
            else:
                equation_string = f"{equation_string} {fixed_parameters[j]} * {X_cols[j]}"

    X_df_test = pd.DataFrame(X_test[X_cols])
    y_actual = pd.DataFrame(y_test[y_lin_log]).values

    y_predict = linear_regression_equation(fixed_parameters, X_df_test.values).astype("float64")

    # %% calculate the MAPE
    # make sure that all equations are on the same scale
    if "log" in y_lin_log:
        y_predict = np.exp(y_predict)
        y_actual = np.exp(y_actual)

    residuals = y_predict - y_actual

    # median absolute percentage error
    absolute_percent_error = np.abs(residuals) / np.maximum(np.abs(y_actual), EPSILON)
    test_mdape = np.median(absolute_percent_error)
    test_mape = np.mean(absolute_percent_error)
    test_rmse = np.sqrt(np.mean((residuals)**2))
    r2 = r2_score(y_actual, y_predict)

    final_equations_df.loc[i, "test_mdape"] = test_mdape
    final_equations_df.loc[i, "test_mape"] = test_mape
    final_equations_df.loc[i, "test_rmse"] = test_rmse
    final_equations_df.loc[i, "equation"] = equation_string

    print(round(test_mdape, 3), i, equation_string)

    # make residual plot
    plt.plot(y_predict, y_actual, 'o')
    plt.xlabel('predicted')
    plt.ylabel('actual')
    plt.title(f"{equation_string}\n MDAPE={round(test_mdape, 3)}, MAPE={round(test_mape, 3)}, RMSE={round(test_rmse,3)}")
    plot_name = f"Residual Plot MDAPE={round(test_mdape, 3)} {equation_string.replace(' ','')}.png"
    plot_name = plot_name.replace("/", "div")
    plt.savefig(fname=os.path.join(plot_directory, plot_name))
    plt.close()

    # make a plot of TDD vs Y
    x_tdd = np.arange(0, 501, 1)
    plot_df = pd.DataFrame(columns=X_cols)
    if "TDD" in str(X_cols):
        plot_df["TDD"] = x_tdd
        plt.plot(X_train["TDD"], y_train[y_lin_log.replace("log_", "")], ".", label="Train Data")
        plt.plot(X_test["TDD"], y_test[y_lin_log.replace("log_", "")], ".", label="Test Data")
        if "log_TDD" in str(X_cols):
            plot_df["log_TDD"] = np.log(x_tdd + 1)
        if "1/TDD" in str(X_cols):
            plot_df["1/TDD"] = 1/(x_tdd + 1)

        for x_cho in [0, 250, 500]:
            for x_bmi in [12, 25, 45]:
                line_name = f"BMI={x_bmi} CHO={x_cho}"
                for x_col in X_cols:
                    if ("TDD" not in x_col) & ("log_TDD" not in x_col):
                        if "X_intercept" in x_col:
                            plot_df["X_intercept"] = 1
                        if "BMI" in str(X_cols):
                            if "BMI" in x_col:
                                plot_df["BMI"] = x_bmi
                            if "log_BMI" in x_col:
                                plot_df["log_BMI"] = np.log(x_bmi + 1)
                        else:
                            line_name = line_name.replace(f"BMI={x_bmi}", "BMI=NA")
                        if "CHO" in str(X_cols):
                            if "CHO" in x_col:
                                plot_df["CHO"] = x_cho
                            if "log_CHO" in x_col:
                                plot_df["log_CHO"] = np.log(x_cho + 1)
                        else:
                            line_name = line_name.replace(f"CHO={x_cho}", "CHO=NA")

                y_estimated = linear_regression_equation(fixed_parameters, plot_df[X_cols].values).astype("float64")
                if "log" in y_lin_log:
                    y_estimated = np.exp(y_estimated)
                plt.plot(x_tdd, y_estimated, label=line_name)

        plt.xlabel("TDD")
        plt.ylabel(y_lin_log.replace("log_",""))
        plt.title(
            f"{equation_string}\n MDAPE={round(test_mdape, 3)}, MAPE={round(test_mape, 3)}, RMSE={round(test_rmse, 3)}"
        )
        plt.legend()
        plot_name = f"TDD Plot MDAPE={round(test_mdape, 3)} {equation_string.replace(' ', '')}.png"
        plot_name = plot_name.replace("/", "div")
        plt.savefig(fname=os.path.join(plot_directory, plot_name))
        plt.close()
    else:
        print("skipping this plot")


# add in Rayhan Special for CIR
# cir_rayhan = (0.39556 * CHO + 62.762) * TDD^(-0.71148)
i = i + 1

final_equations_df.loc[i, ["y", "X_intercept", "BMI", "CHO", "TDD", "beta_1/TDD"]] = [
    "log_CIR_Rayhan",
    "off",
    "off",
    "off",
    "off",
    "off",
]

if "CIR" in file_name:
    # TODO: calculate the validation loss too
    equation_string = "CIR = (0.39556 * CHO + 62.762) * TDD^-0.71148"
    y_predict_cir_rayhan = pd.DataFrame(np.multiply(0.39556 * X_test["CHO"].values + 62.762, X_test["TDD"].values ** (-0.71148))).values
    y_actual = pd.DataFrame(y_test["CIR"]).values

    residuals = y_predict_cir_rayhan - y_actual

    # median absolute percentage error
    absolute_percent_error = np.abs(residuals) / np.maximum(np.abs(y_actual), EPSILON)
    test_mdape = np.median(absolute_percent_error)
    test_mape = np.mean(absolute_percent_error)
    test_rmse = np.sqrt(np.mean((residuals)**2))
    r2 = r2_score(y_actual, y_predict)

    final_equations_df.loc[i, "test_mdape"] = test_mdape
    final_equations_df.loc[i, "test_mape"] = test_mape
    final_equations_df.loc[i, "test_rmse"] = test_rmse
    final_equations_df.loc[i, "equation"] = equation_string

final_equations_df.sort_values("test_mdape", inplace=True)
now = datetime.now().strftime("%m-%d-%y")
final_equations_df.to_csv(f"final_test_results-{now}-{file_name}")
