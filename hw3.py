import numpy as np                     # needed for arrays
from numpy.linalg import solve         # needed for matrices
from read_netlist import read_netlist  # supplied function to read the netlist
import comp_constants as COMP          # needed for the common constants

# this is the list structure that we'll use to hold components:
# [ Type, Name, i, j, Value ]

################################################################################
# How large a matrix is needed for netlist? This could have been calculated    #
# at the same time as the netlist was read in but we'll do it here             #
# Input:                                                                       #
#   netlist: list of component lists                                           #
# Outputs:                                                                     #
#   node_cnt: number of nodes in the netlist                                   #
#   volt_cnt: number of voltage sources in the netlist                         #
################################################################################

def get_dimensions(netlist):
    max_node_number = 0
    max_voltage_count = 0
    for index in range(len(netlist)):
        current_node = max(netlist[index][2], netlist[index][3])
        if current_node > max_node_number:
            max_node_number = current_node
        if netlist[index][0] == 1:
            max_voltage_count += 1
    node_cnt = int(max_node_number)
    volt_cnt = int(max_voltage_count)
    return node_cnt, volt_cnt

################################################################################
# Function to stamp the components into the netlist                            #
# Input:                                                                       #
#   y_add:    the admittance matrix                                            #
#   netlist:  list of component lists                                          #
#   currents: the matrix of currents                                           #
#   node_cnt: the number of nodes in the netlist                               #
# Outputs:                                                                     #
#   node_cnt: the number of rows in the admittance matrix                      #
################################################################################

def stamper(y_add, netlist, currents, num_nodes):
    for comp in netlist:
        i = comp[COMP.I] - 1
        j = comp[COMP.J] - 1

        if comp[COMP.TYPE] == COMP.R:
            if i >= 0:
                y_add[i, i] += 1.0 / comp[COMP.VAL]
            if j >= 0:
                y_add[j, j] += 1.0 / comp[COMP.VAL]
            if i >= 0 and j >= 0:
                y_add[i, j] += -1.0 / comp[COMP.VAL]
                y_add[j, i] += -1.0 / comp[COMP.VAL]
        elif comp[COMP.TYPE] == COMP.VS:
            num_nodes = num_nodes + 1
            M = num_nodes
            if i >= 0:
                y_add[M - 1, i] = 1.0
                y_add[i, M - 1] = 1.0
            if j >= 0:
                y_add[M - 1, j] = -1.0
                y_add[j, M - 1] = -1.0
            currents[M - 1] = comp[COMP.VAL]
            voltage[M - 1] = 0
        elif comp[COMP.TYPE] == COMP.IS:
            if i >= 0:
                if comp[COMP.VAL] >= 0:
                    currents[i] -= 1.0 * comp[COMP.VAL]
                else:
                    currents[i] += 1.0 * comp[COMP.VAL]
            if j >= 0:
                if comp[COMP.VAL] >= 0:
                    currents[j] += 1.0 * comp[COMP.VAL]
                else:
                    currents[j] -= 1.0 * comp[COMP.VAL]

    node_cnt = num_nodes
    return node_cnt


################################################################################
# Start the main program now...                                                #
################################################################################

# Read the netlist
netlist = read_netlist()

# Get the dimensions of the netlist
node_cnt, volt_cnt = get_dimensions(netlist)

print("Total count of nodes and voltages:", node_cnt + volt_cnt)

# Create arrays of right size
y_add = np.zeros((node_cnt + volt_cnt, node_cnt + volt_cnt), dtype=float)
currents, voltage = np.zeros(node_cnt + volt_cnt, dtype=float), np.zeros(node_cnt + volt_cnt, dtype=float)

# Stamp the matrix
node_cnt = stamper(y_add, netlist, currents, node_cnt)

# Solve for the voltages
voltage = np.linalg.solve(y_add, currents)

print(voltage)
