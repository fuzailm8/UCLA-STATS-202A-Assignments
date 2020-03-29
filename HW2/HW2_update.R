############################################################# 
## Stat 202A 2019 Fall - Homework 02
## Author: 
## Date : 
#############################################################

#############################################################
## INSTRUCTIONS: Please fill in the corresponding function. Do not change function names, 
## function inputs or outputs. Do not write anything outside the function. 
## See detailed requirement in python files.
#############################################################

###############################################
## Function 1A                  			 ##
###############################################

sample_uniform <- function(size=10000, low=0, high=1){

  l = c()
  x = 0
  N = 1000
  for (i in 0:N) {
    x = (5*x + 3) %% 128  
    l[i] <- x 
  }
  l = l/128
  return(l)
}

###############################################
## Function 1B                  			 ##
###############################################

sample_normal_chain <- function(x0, c, chain_length=100, mean=0, var=1){
  a = -5
  b = 5
  N = 1000
  c = 1
  
  x = runif(N, min = a, max = b)
  
  x_c = 2*c*runif(N,0,1) + (x - c)
  
  u = runif(N,0,1)
  
  X_out = numeric(N)
  
  for (j in 1:N){
    if( (dnorm(x_c[j])/dnorm(x[j])) > 1)
    {
      X_out[j] = x_c[j] 
    } 
    else {
      if(u[j] < (dnorm(x_c[j])/dnorm(x[j])) ){
        X_out[j] = x_c[j]
      }
      else{
        X_out[j] = x[j]
      }
    }
  }
  
  Xt = rbind(x,X_out)
  
  for (i in 2:99){
    print(paste('Round',i))
    
    x_c = 2*c*runif(N,0,1) + (Xt[i,] - c)
    
    u = runif(N,0,1)
    X_out = numeric(N)
    
    for (j in 1:N){
      if( (dnorm(x_c[j])/dnorm(Xt[i,j])) > 1){
        X_out[j] = x_c[j]}
      else{
        if(u[j] < (dnorm(x_c[j])/dnorm(Xt[i,j])) ){
          X_out[j] = x_c[j] }
        else{
          X_out[j] = Xt[i,j] } 
      } 
    }    
    
    Xt = rbind(Xt, X_out)
  }
  
  print(dim(Xt))
  X = t(Xt)
  print(dim(X)) 
  
    return(X)
}

###############################################
## Function 1C                  			 ##
###############################################

metropolis_simulation <- function(num_chain=1000, chain_length=100, mean=0, var=1){
  
  #list_a = c(0, -0.1, -1, -10) # Add other value as you want.
  #list_b = c(1, 0.1, 1, 10)
  #list_c = c(1, 2)
  
  #To generate specific scenarios
  A = c(-5)
  B = c(5)
  C = c(1)
  
  for (a in A){
    for (b in B){
      for(c in C){
        x0 = runif(num_chain,min=a,max=b)
        X=sample_normal_chain(x0, c, chain_length=100, mean=0, var=1)
        #save_anim(X)
        saveGIF(for(i in 1:chain_length) hist(X[,i], main='Histogram - Metropolis Sampling', xlab= paste('Timestep ',i)), interval=0.2)
      }
    }
  }


}
  
 
###############################################
## Function 2A                  			 ##
###############################################

gibbs_sample <- function(x0, y0, rho, num_chain=1000, chain_length=100, mean=0, var=1){

  x = numeric(100)
  y = numeric(100)

  for (i in 2:100){
    x[i] = rnorm(1)*((1-rho**2)**0.5) + (rho*y[i-1])
    y[i] = rnorm(1)*((1-rho**2)**0.5) + (rho*x[i])
  }

  M = cbind(x,y)
  print(dim(M))

  for (j in 1:999){
    X = numeric(100)
    Y = numeric(100)
    for (i in 2:100){
      X[i] = rnorm(1)*((1-rho**2)**0.5) + (rho*Y[i-1])
      Y[i] = rnorm(1)*((1-rho**2)**0.5) + (rho*X[i]) }
    N = cbind(X,Y)
    M = cbind(M,N)
  }
  X = t(M)
  print(dim(X))
  return(X)
}


# ###############################################
# ## Function 2B                  			 ##
# ###############################################
# 
gibbs_simulation <- function()
  {
  #list_rho = c(0.5, 1, -0.5)
  
  #To generate specific scenarios
  list_rho = c(-0.5)
  x0 = numeric(1000)
  y0 = numeric(1000)
  for (rho in list_rho){
    M = gibbs_sample(x0, y0, rho, num_chain=1000, chain_length=100, mean=0, var=1)
  }
  M_c = M[1:2,]
  M_x = M_c[1,]
  M_y = M_c[2,]
  saveGIF(for (i in 1:100) plot(M_x[1:i],M_y[1:i],type = 'b', main = 'Gibbs Sampler footsteps', xlab = paste('Timestep',i)), interval=0.5)
  
  }
 
# ########################################################
# ## Optional examples (comment out before submitting!) ##
# ########################################################
# 
# ## test()
# 
