import numpy as np

# def factorial(n):
#     if n==1:
#         return 1
#     else:
#         return n*factorial(n-1)

n=int(input('Enter a number whose square root is desired:  '))
guess=int(input('Enter an initial guess:  '))

def babylonian(n, guess):
    e=0.0001
    if abs(guess*guess - n) < e:
        return np.round(guess,2)
    else:
        new_guess=0.5*(guess+(n/guess))
        return babylonian(n , new_guess)

print(f"The square root of {n} is: ", babylonian(n,guess))