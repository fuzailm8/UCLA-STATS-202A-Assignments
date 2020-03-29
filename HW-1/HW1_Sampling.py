# -*- coding: utf-8 -*-
"""

 Stat 202A 2019 Fall - Homework 01
 Author: Fuzail Mujahid Khan (UID: 405428622)
 Date : 10/4/2019

 INSTRUCTIONS: Please fill in the corresponding function. Do not change function names, 
 function inputs or outputs. Do not write anything outside the function. 
 Do not use any of Python's built in functions for matrix inversion or for linear modeling 
 (except for debugging or in the optional examples section).
 
"""

import numpy as np

###############################################
## Function 1: (2a) Uniform 				 ##
###############################################

def sample_uniform(low=0, high=1):
        import matplotlib.pyplot as plt
        l = []
        x = 0
        N = 128
        for i in range(N):
            x = (5*x + 3) % 128  
            l.append(x)
        l = [i/128 for i in l]
        x0 = l[:N-1]
        x1 = l[1:N]
        plt.hist(l, bins = 16)
        plt.title('Histogram of the uniform random numbers by the LCG method X(t)')
        plt.xlabel('Uniform random numbers [0-1]')
        plt.ylabel('Frequency')
        plt.show()
        plt.scatter(x0,x1,s=0.1)
        plt.title('Scatter plot of X(t) vs X(t+1)')
        plt.xlabel('X(t)')
        plt.ylabel('X(t+1)')
        plt.show()
        
sample_uniform()        

###############################################
## Function 2: (2b) Exponential				 ##
###############################################

def sample_exponential(k=1):
        import matplotlib.pyplot as plt
        import math
        x = np.random.uniform(size = 100000)
        x = list(x)
        y = [-math.log(1-i)/k for i in x]
        a,b,c = plt.hist(y, bins = 50, density = True)
        plt.title('Histogram of Exponential(1) distribution using uniform random numbers')
        plt.xlabel('X')
        plt.ylabel('Frequency')          
        plt.show()

###############################################
## Function 3: (2c) Normal	 				 ##
###############################################

def sample_normal(mean=0, var=1):
        import matplotlib.pyplot as plt
        import math
        x = np.random.uniform(size = 100000)
        y = np.random.uniform(size = 100000)

        R = [math.sqrt(-2*math.log(i)) for i in x]
        theta = [2*math.pi*i for i in y]
        X_ = []
        Y_ = []
        for i in range(100000):
            X_.append(R[i]*math.cos(theta[i]))
            Y_.append(R[i]*math.sin(theta[i]))
        plt.scatter(X_,Y_,s=0.5)
        plt.title('Scatter plot of normal(0,1) - X(t) vs Y(t)')
        plt.xlabel('X(t)')
        plt.ylabel('Y(t)')
        plt.show()
        T = [(i**2)/2 for i in R] 
        plt.hist(T, bins = 50)
        plt.title('Histogram of T = (R^2)/2')
        plt.xlabel('T = (R^2)/2')
        plt.ylabel('Frequency')
        plt.show()


###############################################
## Function 4: (3) Monte Carlo 				 ##
###############################################

def monte_carlo(d=2):
        #To estimate pi. The variable 'pi' stores the estimated value of pi 
        N = 100000
        x = np.random.uniform(size=N)
        y = np.random.uniform(size=N)
        U1 = [2*i-1 for i in x]
        U2 = [2*i-1 for i in y]

        count = 0
        for i in range(N):
            if((U1[i])**2 + (U2[i])**2 <= 1):
                count = count + 1
        ratio = count/N
        pi = ratio*4
        print('Estimated value of pi = ',pi)

        #To estimate volume of 5-d unit ball
        N = 100000
        x1 = np.random.uniform(size=N)
        x2 = np.random.uniform(size=N)
        x3 = np.random.uniform(size=N)
        x4 = np.random.uniform(size=N)
        x5 = np.random.uniform(size=N)

        U1 = [2*i-1 for i in x1]
        U2 = [2*i-1 for i in x2]
        U3 = [2*i-1 for i in x3]
        U4 = [2*i-1 for i in x4]
        U5 = [2*i-1 for i in x5]

        count = 0
        for i in range(N):
            if((U1[i])**2 + (U2[i])**2 + (U3[i])**2 + (U4[i])**2 + (U5[i])**2 <= 1):
                count = count + 1
        ratio = count/N
        vol = ratio*(2**5)
        print('Estimated volume of 5 dimensional unit ball is ',vol)

        #To estimate volume of 10-d unit ball
        N = 100000
        x1 = np.random.uniform(size=N)
        x2 = np.random.uniform(size=N)
        x3 = np.random.uniform(size=N)
        x4 = np.random.uniform(size=N)
        x5 = np.random.uniform(size=N)
        x6 = np.random.uniform(size=N)
        x7 = np.random.uniform(size=N)
        x8 = np.random.uniform(size=N)
        x9 = np.random.uniform(size=N)
        x10 = np.random.uniform(size=N)

        U1 = [2*i-1 for i in x1]
        U2 = [2*i-1 for i in x2]
        U3 = [2*i-1 for i in x3]
        U4 = [2*i-1 for i in x4]
        U5 = [2*i-1 for i in x5]
        U6 = [2*i-1 for i in x6]
        U7 = [2*i-1 for i in x7]
        U8 = [2*i-1 for i in x8]
        U9 = [2*i-1 for i in x9]
        U10 = [2*i-1 for i in x10]

        count = 0
        for i in range(N):
            if((U1[i])**2 + (U2[i])**2 + (U3[i])**2 + (U4[i])**2 + (U5[i])**2 + (U6[i])**2 + (U7[i])**2 + (U8[i])**2 + (U9[i])**2 + (U10[i])**2 <= 1):
                count = count + 1
        ratio = count/N
        vol = ratio*(2**10)
        print('Estimated volume of 10 dimensional unit ball is ',vol)

########################################################
## Optional examples (comment out before submitting!) ##
########################################################

## test()

