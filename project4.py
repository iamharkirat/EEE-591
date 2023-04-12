import numpy as np
import shutil
import subprocess
import warnings

warnings.filterwarnings("ignore")
np.seterr(divide='ignore')

# list of all number of fan values
fan = [2, 3, 4, 5, 6, 7]

# list of all number of inverters
inv_number = [3, 5, 7, 9, 11, 13]

# list all combinations of fan and number of inverters (each pair is taken as a tuple)
combinations = [(f, inv) for f in fan for inv in inv_number]

# delay_result is a list to store the delay_results of delay time
delay_result = []

# ascii value of character 'b'
node = 98

for fan_new, inv_new in combinations:
    # copy the header file to a temporary file
    shutil.copyfile('header.sp', 'InvChain_temp.sp')

    with open("InvChain_temp.sp", "a") as file:
        # writing fan parameter line
        line_fan_param = f".param fan = {fan_new}\n"
        file.write(line_fan_param)

        # writing fan lines
        line = f"Xinv1 a {chr(node)} inv M = 1\n"
        file.write(line)

        # write second to last-1 conditions
        for j in np.arange(1, inv_new-1, 1):
            # setting fan multipliers for each stage
            fans = "**" + str(j)
            line = f"Xinv{j + 1} {chr(j + node - 1)} {chr(j + node)} inv M = fan{fans}\n"
            file.write(line)

        # writing the last stage, always ends with node "z",
        # and set the fan multipliers
        fans = "**" + str(j + 1)
        line = f"Xinv{inv_new} {chr(j + node)} z inv M = fan{fans}\n"
        file.write(line)

        # writing end line
        line_end = ".end\n"
        file.write(line_end)

    # overwriting the original InvChain file with InvChain temp file
    shutil.copyfile('InvChain_temp.sp', 'InvChain.sp')

    # running hspice
    pro = subprocess.Popen(["hspice", "InvChain.sp"], stdout=subprocess.PIPE)
    output, err = pro.communicate()

    # extract delay time from csv output
    data = np.recfromcsv("InvChain.mt0.csv", comments="$", skip_header=3)
    delay_result.append(data["tphl_inv"][()])

# after test run is finished, the minimum delay time is found from delay_result list
opt_delay = min(delay_result)

# using minimum delay time to find the corresponding combination
opt_comb = combinations[delay_result.index(opt_delay)]

# printing all combinations of (fan number and num of inverters) and corresponding delay time
print("fan inverter  delay(sec)")
for i, comb in enumerate(combinations):
    print(f"{comb[0]:4d} {comb[1]:8d}  {delay_result[i]:6.4e}")

# print minimum delay time & its corresponding configuration
print(f"\nminimum delay (sec): {opt_delay:.5e}")
print(f"optimum configuration is fan = {opt_comb[0]:d}; inverter = {opt_comb[1]:d}")
