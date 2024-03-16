import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

class Integral:
    def __init__(self,a,b,f,n):
        self.a = a
        self.b = b 
        self.f = f 
        self.n = n 
    
    def compute_integral(self):
        """
        Approximates the definite integral of a function using the trapezoidal rule.
        
        Parameters:
            f : function
                The function to be integrated.
            a : float
                Lower limit of integration.
            b : float
                Upper limit of integration.
            n : int
                Number of subintervals (trapezoids) for the approximation.
        
        Returns:
            float
                Approximation of the definite integral.
        """
        h = (self.b - self.a) / self.n  # Width of each subinterval
        integral = 0.5 * (self.f(self.a) + self.f(self.b))  # Initialize integral with the average of the function at endpoints
        for i in range(1, self.n):
            integral += self.f(self.a + i * h)  # Add value of the function at each interior point
        integral *= h  # Multiply by the width of each subinterval
        return integral

def func_to_integrate(x):
    return (2/np.sqrt(np.pi))*np.exp(-x**2)

class Wall:
    def __init__(self, k, Ti, TB):
        self.k = k  # Thermal diffusivity
        self.Ti = Ti  # Initial temperature
        self.TB = TB  # Boundary temperature at x=0 and t>0
        self.dT = Ti - TB  # Temperature difference

    def func_to_integrate(self,s):
        return (2/np.sqrt(np.pi))*np.exp(-s**2)
    
    def solve_temperature_distribution(self, N, L, t_value):
        x_values = np.linspace(0, L, N)
        x_values = x_values / (2 * np.sqrt(self.k * t_value))
        #T = self.Ti - self.dT * Integral(np.zeros(len(x_values)),x_values,func_to_integrate,100).compute_integral()
        T = self.Ti - self.dT * erfc(x_values)
        #T = self.Ti - self.dT * Integral(x_values,np.ones(len(x_values))*10e+8,func_to_integrate,100000).compute_integral()
        T[0] = self.TB  # Boundary condition at x=0
        return T

    def plot_temperature_distribution(self, T, L):
        N = len(T)
        x_values = np.linspace(0, L, N)
        plt.plot(x_values, T)
        plt.title('Temperature Distribution in the Wall')
        plt.xlabel('Position (x)')
        plt.ylabel('Temperature (T)')
        plt.grid(True)
        plt.show()

# Example usage
k = 0.1   # Thermal diffusivity
Ti = 100.0  # Initial temperature
TB = 20.0   # Boundary temperature at x=0 and t>0
L = 10.0   # Length of the wall
t_final = 1
N = 100   # Number of spatial grid points

wall = Wall(k, Ti, TB)
for time in [0.1,1,10,20,50,100,150,200]:
    temperature_distribution = wall.solve_temperature_distribution(N, L, time)
    wall.plot_temperature_distribution(temperature_distribution, L)
