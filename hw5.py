import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Problem 1: y' = cos(t); initial value: y(0)=1

def dy_dt(y, t):
    return np.cos(t)

t1 = np.linspace(0, 7, 7000)
y0_1 = 1

y1 = odeint(dy_dt, y0_1, t1)

# Problem 2: y' = -y + t**2*np.exp(-2*t) + 10; initial value y(0) = 0

def dy_dt(y, t):
    return -y + t**2*np.exp(-2*t) + 10

t2 = np.linspace(0, 7, 7000)
y0_2 = 0

y2 = odeint(dy_dt, y0_2, t2)

# Problem 3: y''+4y'+4y = 25*cos(t) + 25*sin(t); initial values y(0) = 1, y'(0) = 1

def dy_dt(y, t):
    return [y[1], 25*np.cos(t) + 25*np.sin(t) - 4*y[1] - 4*y[0]]

t3 = np.linspace(0, 7, 7000)
y0_3 = [1, 1]

y3 = odeint(dy_dt, y0_3, t3)

# Plotting the graphs

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10,15))

# Graph 1
axes[0].plot(t1, y1, label="y")
axes[0].set_xlabel('t')
axes[0].set_ylabel('y')
axes[0].set_title("Problem 1: $y' = \cos(t)$, $y(0)=1$")

# Graph 2
axes[1].plot(t2, y2, label="y")
axes[1].set_xlabel('t')
axes[1].set_ylabel('y')
axes[1].set_title("Problem 2: $y' = -y + t^2 e^{-2t} + 10$, $y(0)=0$")

# Graph 3
axes[2].plot(t3, y3[:, 0], label='y')
axes[2].plot(t3, y3[:, 1], label="y'")
axes[2].set_xlabel('t')
axes[2].set_title("Problem 3: $y''+4y'+4y = 25\cos(t) + 25\sin(t)$, $y(0) = 1$, $y'(0) = 1$")
axes[2].legend()

plt.tight_layout()
plt.show()
