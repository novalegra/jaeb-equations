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


def adjusted_r_2(r_2, n, k):
    return 1 - ((1 - r_2) * (n - 1) / (n - k - 1))


def get_coeff(combo_list):
    model_params = combo_list[2:]

    cleaned_model_params = []
    for param in model_params:
        if param != "off":
            cleaned_model_params.append(param)
    return cleaned_model_params


def should_plot(combo_list):
    print(combo_list)
    result = tuple(combo_list) in [
        ("log_BASAL", False, "off", "log_CHO", "log_TDD"),
        ("BASAL", True, "off", "log_CHO", "TDD"),
        ("BASAL", False, "log_BMI", "CHO", "TDD"),
        ("log_BASAL", True, "BMI", "log_CHO", "TDD"),
        ("log_BASAL", True, "off", "off", "log_TDD"),
        ("BASAL", False, "off", "off", "TDD"),
        ("log_BASAL", True, "off", "CHO", "log_TDD"),
        ("log_CIR", True, "off", "CHO", "TDD"),
        ("log_CIR", True, "BMI", "log_CHO", "log_TDD"),
        ("log_CIR", True, "off", "log_CHO", "log_TDD"),
        ("log_CIR", False, "log_BMI", "log_CHO", "TDD"),
        ("log_CIR", False, "off", "log_CHO", "TDD"),
        ("log_CIR", True, "off", "off", "TDD"),
        ("log_CIR", True, "off", "off", "log_TDD"),
        ("CIR", False, "off", "log_CHO", "off"),
        ("log_ISF", True, "log_BMI", "log_CHO", "log_TDD"),
        ("log_ISF", True, "log_BMI", "off", "log_TDD"),
        ("log_ISF", True, "log_BMI", "off", "off"),
        ("log_ISF", True, "off", "off", "log_TDD"),
        ("log_ISF", False, "log_BMI", "log_CHO", "TDD"),
        ("log_ISF", False, "off", "log_CHO", "TDD"),
        ("ISF", False, "off", "log_CHO", "off"),
    ]

    return result


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

# scale the X data so that all have zero mean and std of 1
scaler = StandardScaler().fit(X_train)
X_train_scaled = pd.DataFrame(
    scaler.transform(X_train), columns=X_train.columns, index=X_train.index
)

# keep and uncomment this if you need to save the folds
# # break the training set into 5 folds for model selection with cross validation
# kf = KFold(n_splits=5)
# fold=0
# for i, j in kf.split(X_train):
#     fold = fold + 1
#     X_train_fold = X_train.iloc[i, :]
#     X_val_fold = X_train.iloc[j, :]
#     y_train_fold = y_train.iloc[i, :]
#     y_val_fold = y_train.iloc[j, :]
#     # save the folds (once)
#     X_train_fold.to_csv(os.path.join(data_path, "X_train_fold_{}_2021-06-12_equation_paper_aspirational_data_reduced.csv".format(fold)))
#     X_val_fold.to_csv(os.path.join(data_path, "X_val_fold_{}_2021-06-12_equation_paper_aspirational_data_reduced.csv".format(fold)))
#     y_train_fold.to_csv(os.path.join(data_path, "y_train_fold_{}_2021-06-12_equation_paper_aspirational_data_reduced.csv".format(fold)))
#     y_val_fold.to_csv(os.path.join(data_path, "y_val_fold_{}_2021-06-12_equation_paper_aspirational_data_reduced.csv".format(fold)))


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

    for pm in ["const", "BMI", "CHO", "TDD"]:
        ac_df["{}_scaled".format(pm)] = np.nan
    for pm in ["log_BMI", "log_CHO", "log_TDD"]:
        ac_df["{}_scaled".format(pm)] = np.nan

    for metric in [
        "MAPE",
        "MAE",
        "RMSE",
        "MAX_ERROR",
        "EXPLAIN_VAR",
        "R2",
        "ADJ_R2",
        "model_warning",
    ]:
        ac_df["{}_mean".format(metric)] = np.nan
    for metric in ["MAPE", "MAE", "RMSE", "MAX_ERROR", "model_warning"]:
        ac_df["{}_max".format(metric)] = np.nan

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

            # let's build the full model with all training data so we can look at AIC, BIC and coefficients
            # fit with huber & grab coefficients/intercept
            huber_regr = linear_model.HuberRegressor(fit_intercept=fit_intercept)
            huber_regr.fit(X_train[X_cols], np.ravel(y_train[y_lin_log]))

            if fit_intercept:
                ac_df.loc[combo, "intercept_huber"] = huber_regr.intercept_
            for i, key in enumerate(get_coeff(list(ac))):
                ac_df.loc[combo, "{}_huber".format(key)] = huber_regr.coef_[i]

            # fit with huber, scaled, & grab coefficents/intercept
            huber_regr_scaled = linear_model.HuberRegressor(fit_intercept=fit_intercept)
            huber_regr_scaled.fit(X_train_scaled[X_cols], np.ravel(y_train[y_lin_log]))

            if fit_intercept:
                ac_df.loc[
                    combo, "intercept_huber_scaled"
                ] = huber_regr_scaled.intercept_
            for i, key in enumerate(get_coeff(list(ac))):
                ac_df.loc[
                    combo, "{}_scaled_huber".format(key)
                ] = huber_regr_scaled.coef_[i]

            # fit with stats model
            if fit_intercept:
                stats_mod_scaled = sm.OLS(
                    y_train[y_lin_log], sm.add_constant(X_train_scaled[X_cols])
                )
            else:
                stats_mod_scaled = sm.OLS(y_train[y_lin_log], X_train_scaled[X_cols])

            res_scaled = stats_mod_scaled.fit()
            for idx in res_scaled.params.index:
                ac_df.loc[combo, "{}_scaled".format(idx)] = res_scaled.params[idx]
                print(res_scaled.params[idx])

            if fit_intercept:
                stats_mod = sm.OLS(y_train[y_lin_log], sm.add_constant(X_train[X_cols]))
            else:
                stats_mod = sm.OLS(y_train[y_lin_log], X_train[X_cols])

            res = stats_mod.fit()

            ac_df.loc[combo, "n_params"] = len(X_cols) + (fit_intercept * 0.5)
            ac_df.loc[combo, "summary_all"] = res.summary()
            if "[2]" in str(res.summary()):
                has_warning = True
            else:
                has_warning = False
            ac_df.loc[combo, "model_warning"] = has_warning
            ac_df.loc[combo, "aic"] = res.aic
            ac_df.loc[combo, "bic"] = res.bic

            sm_df = pd.DataFrame(res.params, columns=["coef"])
            ci = pd.DataFrame(res.conf_int())
            sm_df["ci_lb"] = ci.loc[:, 0]
            sm_df["ci_ub"] = ci.loc[:, 1]
            sm_df["p_val"] = res.pvalues
            ac_df.loc[combo, "perc_pval_lt05"] = np.sum(res.pvalues < 0.05) / len(
                res.pvalues
            )

            temp_df = sm_df.stack().reset_index()
            cols = list(temp_df["level_0"] + "_" + temp_df["level_1"])
            for c, col in enumerate(cols):
                ac_df.loc[combo, col] = temp_df.loc[c, 0]

            if should_plot(list(ac)):
                model_norm_residuals = res.get_influence().resid_studentized_internal
                model_cooks = res.get_influence().cooks_distance[0]

                outlier_threshold = np.mean(model_cooks) * 3
                outliers = np.argsort(model_cooks)[model_cooks > outlier_threshold]
                print(len(outliers), outliers)

                QQ = ProbPlot(model_norm_residuals)
                plot_lm_2 = QQ.qqplot(line="45", alpha=0.5, color="#4C72B0", lw=1)
                plot_lm_2.axes[0].set_title(
                    f"Normal Q-Q for {list(ac)}\nOutliers: {len(outliers)}/{len(model_cooks)} ({round(len(outliers)/len(model_cooks)* 100)}%)"
                )
                plot_lm_2.axes[0].set_xlabel("Theoretical Quantiles")
                plot_lm_2.axes[0].set_ylabel("Standardized Residuals")

                combo_description = "_".join(str(item) for item in list(ac))
                title = f"{pathlib.Path()}/plots/normal_qq_{combo_description}.png"

                plt.savefig(title)

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

                if len(X_cols) == 1:
                    X_train_X_cols = X_train_fold[X_cols].values.reshape(-1, 1)
                    X_val_X_cols = X_val_fold[X_cols].values.reshape(-1, 1)
                else:
                    X_train_X_cols = X_train_fold[X_cols]
                    X_val_X_cols = X_val_fold[X_cols]

                y_train_data = y_train_fold[y_lin_log].values.reshape(-1, 1)
                y_val_data = y_val_fold[y_lin_log].values.reshape(-1, 1)

                # regr = linear_model.LinearRegression(fit_intercept=fit_intercept)
                regr = linear_model.HuberRegressor(fit_intercept=fit_intercept)
                regr.fit(X_train_X_cols, np.ravel(y_train_data))
                y_predict = regr.predict(X_val_X_cols)
                if "log" in y_lin_log:
                    y_val_data = np.exp(y_val_data)
                    y_predict = np.exp(y_predict)

                ac_df.loc[
                    combo, "MAPE_fold{}".format(fold)
                ] = mean_absolute_percentage_error(y_val_data, y_predict)
                ac_df.loc[combo, "MAE_fold{}".format(fold)] = median_absolute_error(
                    y_val_data, y_predict
                )
                ac_df.loc[combo, "RMSE_fold{}".format(fold)] = mean_squared_error(
                    y_val_data, y_predict, squared=False
                )
                ac_df.loc[combo, "MAX_ERROR_fold{}".format(fold)] = max_error(
                    y_val_data, y_predict
                )
                ac_df.loc[
                    combo, "EXPLAIN_VAR_fold{}".format(fold)
                ] = explained_variance_score(y_val_data, y_predict)

                R2 = r2_score(y_val_data, y_predict)
                ac_df.loc[combo, "R2_fold{}".format(fold)] = R2
                ac_df.loc[combo, "ADJ_R2_fold{}".format(fold)] = adjusted_r_2(
                    R2, len(y_train_fold), len(X_cols) + (fit_intercept * 1)
                )

                # fit with stats model
                if fit_intercept:
                    stats_mod = sm.OLS(
                        y_train_fold[y_lin_log], sm.add_constant(X_train_fold[X_cols])
                    )
                else:
                    stats_mod = sm.OLS(y_train_fold[y_lin_log], X_train_fold[X_cols])
                res = stats_mod.fit()
                ac_df.loc[combo, "summary_fold{}".format(fold)] = res.summary()
                if "[2]" in str(res.summary()):
                    has_warning = True
                else:
                    has_warning = False
                ac_df.loc[combo, "model_warning_fold{}".format(fold)] = has_warning

                sm_df = pd.DataFrame(res.params, columns=["coef"])
                ci = pd.DataFrame(res.conf_int())
                sm_df["ci_lb"] = ci.loc[:, 0]
                sm_df["ci_ub"] = ci.loc[:, 1]
                sm_df["p_val"] = res.pvalues

                temp_df = sm_df.stack().reset_index()
                cols = list(temp_df["level_0"] + "_" + temp_df["level_1"])
                for c, col in enumerate(cols):
                    ac_df.loc[combo, "{}_fold{}".format(col, fold)] = temp_df.loc[c, 0]

                asdf = 3

            for metric in [
                "MAPE",
                "MAE",
                "RMSE",
                "MAX_ERROR",
                "EXPLAIN_VAR",
                "R2",
                "ADJ_R2",
                "model_warning",
            ]:
                if "{}_fold1".format(metric) in ac_df.columns:
                    ac_df.loc[combo, "{}_mean".format(metric)] = np.mean(
                        [
                            ac_df.loc[combo, "{}_fold1".format(metric)],
                            ac_df.loc[combo, "{}_fold2".format(metric)],
                            ac_df.loc[combo, "{}_fold3".format(metric)],
                            ac_df.loc[combo, "{}_fold4".format(metric)],
                            ac_df.loc[combo, "{}_fold5".format(metric)],
                        ]
                    )
            for metric in ["MAPE", "MAE", "RMSE", "MAX_ERROR", "model_warning"]:
                if "{}_fold1".format(metric) in ac_df.columns:
                    ac_df.loc[combo, "{}_max".format(metric)] = np.max(
                        [
                            ac_df.loc[combo, "{}_fold1".format(metric)],
                            ac_df.loc[combo, "{}_fold2".format(metric)],
                            ac_df.loc[combo, "{}_fold3".format(metric)],
                            ac_df.loc[combo, "{}_fold4".format(metric)],
                            ac_df.loc[combo, "{}_fold5".format(metric)],
                        ]
                    )

    ac_df.sort_values(
        by=["model_warning_max", "perc_pval_lt05", "MAPE_mean"],
        ascending=[True, False, True],
        inplace=True,
    )
    ac_df.reset_index(inplace=True)
    ac_df.to_csv("{}-equation-results-MAPE.csv".format(y[0]))
