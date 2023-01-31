import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def equation(a,b,c,x):  # Integral function ax^2 + bx + c
    return a*x**2 + b*x +c

def integrate(constants):
    r = np.linspace(0, 5, 100)
    
    # Loop over list of constants
    for i, constant in enumerate(constants):
        a, b, c = constant
        
        # Calculate the integral for each value of r
        integral = [quad(equation, 0, y, args=(a, b, c))[0] for y in r]
        
        # plot the results of integration
        plt.plot(r, integral, label=f'Polynomial {i+1}')

    # Add title, x&y axis labels, and legend    
    plt.title('Integral of ax^2 + bx + c')
    plt.xlabel('r')
    plt.ylabel('Integral Values')
    plt.legend()
    plt.show()

constants = [(2, 3, 4), (2, 1, 1)]
integrate(constants)


