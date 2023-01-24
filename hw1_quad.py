import numpy as np
import math as m
import cmath

a=int(input('Input Coefficient a: '))
b=int(input('Input Coefficient b: '))
c=int(input('Input Coefficient c: '))

root=(m.pow(b,2))-(4*a*c)

if abs(root) < 1e-10: 
    print("Double Root")

elif root > 0:
    d=np.sqrt(root)
    sol_1= ((-b)+d)/(2*a)
    sol_2= ((-b)-d)/(2*a)
    print(f'Root 1: {sol_1}')
    print(f'Root 2: {sol_2}')

elif root == 0:
    sol=(-b)/(2*a)
    print(f'Double Root: {sol}')
    
elif root < 0:
    d=cmath.sqrt(root)
    sol_1= ((-b)+d)/(2*a)
    sol_2= ((-b)-d)/(2*a)
    print(f'Root 1: {sol_1}')
    print(f'Root 2: {sol_2}')