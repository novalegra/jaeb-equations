# Jaeb Data Equation Exploration

#### -- Project Status: Complete
#### -- Project Disclaimer: This work is for exploration

## Project Objective
The purpose of this project is to create equations based on Jaeb study data that can assist with determining initial Loop settings.

## Definition of Done
This phase of the project will be done when equations have been created for basal rate, carb ratio, and insulin sensitivity factor.

## Project Description
This project first uses Jaeb study data to create an 'aspirational' cohort of subjects that have normal BMI, meet the international consensus on time in range (≥90% CGM), and have bolus and carbohydrate data available. This is done in `process_data_for_fitting.py`.

 The project then uses 70% of the aspirational data to fit the different equation form combinations of BMI, daily CHO, and TDD for basal rate, ISF, and ICR. A 5-fold cross-validation is used to compute value loss. This is done in `manual_folds_with_custom_loss.py`.

 Finally, the project uses the remaining reserved 30% of the aspirational data to evaluate the final equations. This is done in `test_final_models_from_manual_folds_with_custom_loss.py`. 

### Technologies
* Python
* [Anaconda](https://www.anaconda.com/) for our virtual environments
* Pandas for working with data
* Black for code style
* Numpy docstring format

## Getting Started with the Conda Virtual Environment
1. Install [Miniconda](https://conda.io/miniconda.html). CAUTION for python virtual env users: Anaconda will automatically update your .bash_profile
so that conda is launched automatically when you open a terminal. You can deactivate with the command `conda deactivate`
or you can edit your bash_profile.
2. If you are new to [Anaconda](https://docs.anaconda.com/anaconda/user-guide/getting-started/)
check out their getting started docs.
3. If you want the pre-commit githooks to install automatically, then following these
[directions](https://pre-commit.com/#automatically-enabling-pre-commit-on-repositories).
4. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
5. In a terminal, navigate to the directory where you cloned this repo.
6. Run `conda update -n base -c defaults conda` to update to the latest version of conda
7. Run `conda env create -f conda-environment.yml --name [input-your-env-name-here]`. This will download all of the package dependencies
and install them in a conda (python) virtual environment. (Insert your conda env name in the brackets. Do not include the brackets)
8. Run `conda env list` to get a list of conda environments and select the environment
that was created from the environmental.yml file (hint: environment name is at the top of the file)
9. Run `conda activate <conda-env-name>` or `source activate <conda-env-name>` to start the environment.
10. If you did not setup your global git-template to automatically install the pre-commit githooks, then
run `pre-commit install` to enable the githooks.
11. Run `deactivate` to stop the environment.

## Maintaining Compatability with venv and virtualenv
This may seem counterintuitive, but when you are loading new packages into your conda virtual environment,
load them in using `pip`, and export your environment using `pip-chill > requirements.txt`.
We take this approach to make our code compatible with people that prefer to use venv or virtualenv.
This may also make it easier to convert existing packages into pypi packages. We only install packages directly
in conda using the conda-environment.yml file when packages are not available via pip (e.g., R and plotly-orca).

## Getting Started with this project
1. The data was obtained from the [Jaeb observational study](https://docs.google.com/document/d/1rBB__nNhbIt1-HO2mjKT34Ad8KRDWZCY9yvrHSVUxDs/edit#:~:text=https%3A//www.liebertpub.com/doi/10.1089/dia.2020.0535)
2. Data processing/transformation scripts are being kept in the `src` folder of this repo.
