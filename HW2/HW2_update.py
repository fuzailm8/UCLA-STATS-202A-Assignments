# -*- coding: utf-8 -*-
"""

 Stat 202A 2019 Fall - Homework 02
 Author: 
 Date : 

 INSTRUCTIONS: Please fill in the corresponding function. Do not change function names, 
 function inputs or outputs. Do not write anything outside the function.
 
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm
from matplotlib.animation import PillowWriter

### Part 1 : Metropolis Algorithm

def sample_uniform(size=10000, low=0, high=1):

        l = []
        x = 0
        N = size
        for i in range(N):
            x = (5*x + 3) % 128  
            l.append(x)
        l = [i/128 for i in l]
        X = l
        return X

def sample_normal_chain(x0, c, chain_length=100, mean=0, var=1):

        """
        Function 1B: Normal by Metropolis
        Detail :        This function return multiple chains by Metropolis sampling from N(mean, var).
                                For every train, the proposal distribution at x is y ~ Uniform[x-c, x+c].
                                The input x0 is a 1 dimension np.ndarray. 
                                Return a np.ndarray X with size x0.shape[0] * chain_length. In X, each row X[i]
                                should be a chain and t-th column X[:, t] corresponding to all different X_t. 
        """

        import numpy as np
        
        a = -5
        b = 5
        N = 1000
        c = 0.5

        x = np.random.uniform(a,b,size=N)

        x_c = 2*c*np.random.uniform(0,1,size=N) + (x - c)
            
        u = np.random.uniform(size=N)

        X_out = np.zeros(N)

        for j in range(N):
            if( (norm.pdf(x_c[j])/norm.pdf(x[j])) > 1):
                X_out[j] = x_c[j]
            else:
                if(u[j] < (norm.pdf(x_c[j])/norm.pdf(x[j])) ):
                    X_out[j] = x_c[j]
                else:
                    X_out[j] = x[j]
    

        Xt = np.vstack((x, X_out))

        for i in range(1,99):
            print('Round ',i)
            x_c = 2*c*np.random.uniform(0,1,size=N) + (Xt[i] - c)
    
            u = np.random.uniform(size=N)
            X_out = np.zeros(N)
    
            for j in range(N):
                if( (norm.pdf(x_c[j])/norm.pdf(Xt[i][j])) > 1):
                    X_out[j] = x_c[j]
                else:
                    if(u[j] < (norm.pdf(x_c[j])/norm.pdf(Xt[i][j])) ):
                        X_out[j] = x_c[j]
                    else:
                        X_out[j] = Xt[i][j]
                
            Xt = np.vstack((Xt, X_out))
            
        print('Shape of Xt = ',Xt.shape)    
        X = np.transpose(Xt)
        print('Shape of X = ',X.shape) 
      
        return X


def metropolis_simulation(num_chain=1000, chain_length=100, mean=0, var=1):

        """
        Function 1C: Simulate metropolis with different setting.
        Detail :        Try different setting and output movie of histgrams.
        """

        list_a = [0, -0.1, -1, -10] # Add other value as you want.
        list_b = [1, 0.1, 1, 10]
        list_c = [1, 2]

        #for a, b, c in [(a, b, c) for a in list_a for b in list_b for c in list_c]:

        A = [-5]
        B = [5]
        C = [0.5]
        # to test working of algorithm for specific case and see output GIF. You can use the lists provided above too but will take long run time.
        for a, b, c in [(a, b, c) for a in A for b in B for c in C]:
                
                # Sample num_chain x0 from uniform[a,b]
                x0 = np.random.uniform(a,b,size=num_chain)

                # Run Metropolis
                X_normal = sample_normal_chain(x0, c, chain_length, mean, var)
                Xt = np.transpose(X_normal)
                
                # Plot movie and save 
                # Here plot chain_length graphs, each of them is a histogram of num_chain point. 
                # You may use matplotlib.animation and matplotlib.rc to save graphs into gif movies.
        
        def update_hist(num, Xt):
            label = 'timestep {0}'.format(num)
            plt.cla()
            plt.xlabel(label) 
            plt.hist(Xt[num], bins = 20)

        fig = plt.figure()

        anim = animation.FuncAnimation(fig, update_hist, 100, fargs=(Xt, ) )
        writer = PillowWriter(fps=10)
        #anim.save('C:/Users/Dell/Desktop/a_-5_b_5_c_0_5_new_1.gif', dpi=80, writer = writer)
        plt.show()


### Part 2 : Gibbs Sampling

def gibbs_sample(x0, y0, rho, num_chain=1000, chain_length=100, mean=0, var=1):

        """
        Function 2A: Bivariate normal with correlation rho
        Detail :        This function return multiple chains by Gibbs sampling
                                The input x0, y0, rho, num_chain is a number. This time, we use same starting point. 
                                Return a np.ndarray X with size num_chain * chain_length * 2. In X, each row X[i]
                                should be a chain and t-th column X[:, t] corresponding to all different pair (X_t, Y_t). 
        """

        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from scipy.stats import norm
        from matplotlib.animation import PillowWriter


        x = np.zeros(100)
        y = np.zeros(100)

        for i in range(1,100):
            x[i] = np.random.normal()*((1-rho**2)**0.5) + (rho*y[i-1])
            y[i] = np.random.normal()*((1-rho**2)**0.5) + (rho*x[i])

        M = np.transpose(np.array([x,y]))
        print(M.shape)

        for j in range(999):
            X = np.zeros(100)
            Y = np.zeros(100)
            for i in range(1,100):
                    X[i] = np.random.normal()*((1-rho**2)**0.5) + (rho*Y[i-1])
                    Y[i] = np.random.normal()*((1-rho**2)**0.5) + (rho*X[i])
            N = np.transpose(np.array([X,Y]))
            M = np.hstack((M,N))
            #print(M.shape)
            
            
        P = np.reshape(M, (100,1000,-1))
        X = np.reshape(M, (num_chain, chain_length,-1))
        
        return X

def gibbs_simulation():

        """
        Function 2B: Simulate Gibbs with different rho and plot
        Detail :        Try different setting and output movie of histgrams. 
                                Discard first 50 steps and output 50~100 steps only.
        """

        #list_rho = [0, -1, 1] # Add other value as you want.
        list_rho = [0.6]
        
        for rho in list_rho:

                # Run Gibbs Sampling
                x0 = np.zeros(1000)
                y0 = np.zeros(1000)
                
                X = gibbs_sample(x0, y0, rho, num_chain=1000, chain_length=100, mean=0, var=1)
                # Plot movie and save

        print(X.shape)

### The below code is for making the GIFs


# This is for creating scatter plot which plots 1000 points in each of the 100 time steps. 

        def update_scatter(num,X):
            label = 'timestep {0}'.format(num)
            plt.cla()
            plt.xlabel(label)
            chain = X[:,num]
            print(num)
            plt.scatter(chain[:,0],chain[:,1],s=1)

        fig1 = plt.figure()

        anim1 = animation.FuncAnimation(fig1, update_scatter, 100, fargs=(X, ) )
        writer = PillowWriter(fps=10)
##      anim1.save('C:/Users/Dell/Desktop/rho_+0_2_each_scatter_plot.gif', dpi=80, writer = writer)

        plt.show()
        plt.close()


# This is for creating combined scatter plots with 1000 points for t=1, 2000 points for t=2 and so on.
# For t=100, you can see all the 1000 x 100 points on one graph.

        def update_scatter_total(num,X):
            label = 'timestep {0}'.format(num)
            plt.cla()
            plt.xlabel(label)
            t = X[:,:num]
            X_coord = []
            Y_coord = []
            for i in range(num):
                f = t[:,i]
                X_coord.append(f[:,0])
                Y_coord.append(f[:,1])
            print(len(X_coord),len(Y_coord))    
            plt.scatter(X_coord, Y_coord,s=1)    

        fig2 = plt.figure()

        anim2 = animation.FuncAnimation(fig2, update_scatter_total, 100, fargs=(X, ) )
        writer = PillowWriter(fps=10)
##      anim2.save('C:/Users/Dell/Desktop/rho_+0_2_total_scatter_plot.gif', dpi=80, writer = writer)

        plt.show()
        plt.close()


#This is for taking 1 chain and plotting the footsteps of that chain.

        def update_footsteps(num,X):
            label = 'timestep {0}'.format(num)
            plt.cla()
            plt.xlabel(label)
            c = X[0,:]
            print(num)
            x_c = c[:num,0]
            y_c = c[:num,1]
            plt.plot(x_c,y_c,'-o',markersize=1)
            
##        fig3 = plt.figure()

##        anim3 = animation.FuncAnimation(fig3, update_footsteps, 100, fargs=(X, ) )
##        writer = PillowWriter(fps=10)
##        anim3.save('C:/Users/Dell/Desktop/footsteps_check_1.gif', dpi=80, writer = writer)

##        plt.show()

                
gibbs_simulation()
