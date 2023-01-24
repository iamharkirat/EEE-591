import matplotlib.pyplot as plt
from scipy.integrate import quad
import numpy as np

def integral(constants,x_min,x_max):
    x = np.linspace(x_min, x_max, 100)
    plt.figure()
    for i,constant in enumerate(constants):
        a,b,c=constant
        y = a*x**2 + b*x +c
        plt.plot(x,y)

    plt.title('Definite integral of ax^2 + bx + c')
    plt.xlabel('Integration Range')
    plt.ylabel('Itegral Values')
    plt.legend([f'Polynomial {i+1}' for i in range(len(constants))])
    plt.show()

constants=[(2, 3, 4),(2,1,1)]
integral(constants, 0, 5)