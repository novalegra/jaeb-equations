import equation_utils
import utils
from pumpsettings import PumpSettings
from test_equations import run_equation_testing

input_file = "processed-aspirational_final_test_2021_02_04_22-v0_1-4d1a82f"

jaeb = PumpSettings(
    equation_utils.jaeb_basal_equation,
    equation_utils.jaeb_isf_equation,
    equation_utils.jaeb_icr_equation,
)

traditional = PumpSettings(
    equation_utils.traditional_basal_equation,
    equation_utils.traditional_isf_equation,
    equation_utils.traditional_icr_equation,
)

# This will output the results to a file
run_equation_testing(input_file, jaeb, traditional)
