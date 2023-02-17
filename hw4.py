
import PySimpleGUI as sg  # get the higher-level GUI package
import numpy as np  # get numpy
import matplotlib.pyplot as plt  # get matplotlib for plotting

# Constants
YEARS = 70  # number of years we want to live to and have money
ANALYSIS = 10  # how many times would we like to run our calcs
PERCENT_CONVERTER = 100  # used during calcs when we divided by 100

# Define the field names and their indices
FN_MEAN_RETURN_PERCENT = 'Mean Return (%)'
FN_STD_DEV_RETURN_PERCENT = 'Std Dev Return (%)'
FN_YEARLY_CONTRIBUTION = 'Yearly Contribution ($)'
FN_NO_OF_YEARS_OF_CONTRIBUTION = 'No. of Years of Contribution'
FN_NO_OF_YEARS_TO_RETIREMENT = 'No. of Years to Retirement'
FN_ANNUAL_SPEND_IN_RETIREMENT = 'Annual Spend in Retirement'
FN_RETIREMENT = 'CALCULATING ...'

# Placing our names in a list
FIELD_NAMES = [FN_MEAN_RETURN_PERCENT, FN_STD_DEV_RETURN_PERCENT, FN_YEARLY_CONTRIBUTION, \
               FN_NO_OF_YEARS_OF_CONTRIBUTION, FN_NO_OF_YEARS_TO_RETIREMENT, FN_ANNUAL_SPEND_IN_RETIREMENT]
F_MEAN_RETURN_PERCENT = 0  # index for Mean Return (%)
F_STD_DEV_RETURN_PERCENT = 1  # index for Std Dev Return (%)
F_YEARLY_CONTRIBUTION = 2  # index for Yearly Contribution ($)
F_NO_OF_YEARS_OF_CONTRIBUTION = 3  # index for No. of Years of Contribution
F_NO_OF_YEARS_TO_RETIREMENT = 4  # index for No. of Years to Retirement
F_ANNUAL_SPEND_IN_RETIREMENT = 5  # index for Annual Spend in Retirement
F_RETIREMENT = 6  # index for Retirement
NUM_FIELDS = 6  # how many fields there are
B_CALCULATE = 'Calculate'  # the calculate button
B_QUIT = 'Quit'  # the quit button

wealth_store_matrix = np.zeros((YEARS, ANALYSIS), dtype=float)  # create our store matrix

def wealth_calculation(window, entries):
    y_in = float(entries[FN_YEARLY_CONTRIBUTION])
    y_out = float(entries[FN_ANNUAL_SPEND_IN_RETIREMENT])
    r = float(entries[FN_MEAN_RETURN_PERCENT])
    total_contribution_years = int(entries[FN_NO_OF_YEARS_OF_CONTRIBUTION])
    total_years_to_retirement = int(entries[FN_NO_OF_YEARS_TO_RETIREMENT])
    sigma = float(entries[FN_STD_DEV_RETURN_PERCENT])

    wealth_store_matrix = np.zeros((YEARS, ANALYSIS))
    plt.figure()

    for index_analysis in range(ANALYSIS):
        current_val = 0
        last_index = 0
        noise = (sigma / PERCENT_CONVERTER) * np.random.randn(YEARS)

        for year in range(YEARS):
            if year < total_contribution_years:
                current_val = current_val * (1 + (r / PERCENT_CONVERTER) + noise[year]) + y_in
            elif year < total_years_to_retirement:
                current_val = current_val * (1 + (r / PERCENT_CONVERTER) + noise[year])
            else:
                current_val = current_val * (1 + (r / PERCENT_CONVERTER) + noise[year]) - y_out

            if current_val >= 0:
                wealth_store_matrix[year, index_analysis] = current_val
                last_index = year
            else:
                break

        plt.plot(range(last_index + 1), wealth_store_matrix[0:last_index + 1, index_analysis], '-x')

    plt.title('Wealth Over 70 Years ($)')
    plt.ylabel('Wealth ($)')
    plt.xlabel('Years')
    plt.grid(True)
    plt.show()

    total_sum = np.sum(wealth_store_matrix[total_years_to_retirement + 1, :], axis=0)
    return total_sum / ANALYSIS


def update_layout(num_fields, field_names, money):
    # Set up the font and size
    sg.set_options(font=('Helvetica', 20))
    # Start with an empty layout list
    layout = []
    # For each field to create, append a Text and InputText element to the layout list
    for index in range(num_fields):
        layout.append([sg.Text(field_names[index] + ': ', size=(30, 1)),
                       sg.InputText(key=field_names[index], size=(30, 1))])
    # Add an output text box to the layout list
    calc_tot = 'Enter info then double click Calculate...' if money == 0 else str(money)
    layout.append([sg.Text(calc_tot, key='-OUTPUT-', size=(30, 1))])
    # Add two buttons to the layout list
    layout.append([sg.Button(B_CALCULATE), sg.Button(B_QUIT)])
    # Create a window object with the updated layout list
    window = sg.Window('70 Year Wealth Calculator', layout)
    # Read the event and values from the window
    event, values = window.read()
    return window

def create_layout(NUM_FIELDS, FIELD_NAMES, money):
    sg.set_options(font=('Helvetica', 20))  # set up the font and size
    calc_tot = str(money) if money != 0 else 'Enter info then double click Calculate...'
    layout = [[sg.Text(name + ': ', size=(30, 1)), sg.InputText(key=name, size=(30, 1))] for name in FIELD_NAMES]
    layout.append([sg.Text(calc_tot, key='-OUTPUT-', size=(30, 1))])
    layout.append([sg.Button(B_CALCULATE), sg.Button(B_QUIT)])
    return layout

# First call for the update layer
window = sg.Window('70 Year Wealth Calculator', create_layout(NUM_FIELDS, FIELD_NAMES, 0))

# Run the event loop
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == B_QUIT:
        break
    if event == B_CALCULATE:
        money_avg = wealth_calculation(window, values)
        money_avg_with_format = 'Wealth at retirement: ${:,}'.format(int(money_avg))
        window['-OUTPUT-'].update(money_avg_with_format)
    else:
        print("Not Good")
window.close()
