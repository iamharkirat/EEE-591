import numpy as np
from scipy.integrate import quad

# Define a function to substitute '1+x = u' 
def integrate(u):
    return 1/(u*np.sqrt(u-1))

result, error = quad(integrate, 1, np.inf)

# Round result to 8 decimal places
result = np.round(result,8)

# Print np.pi
print('Pi is', np.round(result,8))

# Difference from our output
difference = np.pi-result
print('Difference from numpy.pi is:', np.round(difference, 15))
