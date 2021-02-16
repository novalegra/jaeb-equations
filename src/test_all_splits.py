import equation_utils
import utils
from pumpsettings import PumpSettings
from test_equations import run_equation_testing

directory_path = "/Users/annaquinlan/Desktop/Projects/TP/jaeb-analysis/.reports/processed-30-min-win_aggregated_rows_per_patient_2021_02_04_22-v0_1-4d1a82f/evaluate-equations-2021_02_04_22-v0_1-4d1a82f"
num_splits = 5
group = utils.DemographicSelection.OVERALL

for file_number in range(1, num_splits + 1):
    matching_key = "test_" + str(file_number) + "_" + group.name.lower()
    matching_name = utils.find_matching_file_name(matching_key, ".csv", directory_path)

    jaeb = PumpSettings(
        equation_utils.jaeb_basal_equation,
        equation_utils.jaeb_isf_equation,
        equation_utils.jaeb_icr_equation,
    )

    traditional_fitted = PumpSettings(
        equation_utils.traditional_basal_equation,
        equation_utils.traditional_isf_equation,
        equation_utils.traditional_icr_equation,
    )

    traditional_constants = PumpSettings(
        equation_utils.traditional_constants_basal_equation,
        equation_utils.traditional_constants_isf_equation,
        equation_utils.traditional_constants_icr_equation,
    )

    # This will output the results to a file
    run_equation_testing(matching_name, jaeb, traditional_fitted, traditional_constants)
    print("Finished equation testing for split {}".format(file_number))

