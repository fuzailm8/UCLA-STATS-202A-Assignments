############################################################# 
## Stat 202A - Homework 3
## Author: Fuzail Mujahid Khan  
## Date : 18/10/2019
## Description: This script implements linear regression 
## using the sweep operator
#############################################################

#############################################################
## INSTRUCTIONS: Please fill in the missing lines of code
## only where specified. Do not change function names, 
## function inputs or outputs. You can add examples at the
## end of the script (in the "Optional examples" section) to 
## double-check your work, but MAKE SURE TO COMMENT OUT ALL 
## OF YOUR EXAMPLES BEFORE SUBMITTING.
##
## Very important: Do not use the function "setwd" anywhere
## in your code. If you do, I will be unable to grade your 
## work since R will attempt to change my working directory
## to one that does not exist.
##
## Do not use the following functions for this assignment,
## except when debugging or in the optional examples section:
## 1) lm()
## 2) solve()
#############################################################

 
################################
## Function 1: Sweep operator ##
################################

mySweep <- function(A, k){
  
  # Perform a SWEEP operation on A with the pivot element A[k,k].
  # 
  # A: a square matrix.
  # m: the pivot element is A[k, k].
  # Returns a swept matrix B (which is k by k).
  
  #############################################
  ## FILL IN THE BODY OF THIS FUNCTION BELOW ##
  #############################################

  B = A 
  K = k
  k = 0 
  
  nrows = dim(A)[1]
  ncols = dim(A)[2]
  
  for (k in 1:K) { 
    for (i in 1:nrows) {
      for (j in 1:ncols) { 
        if(i == k & j == k) 
          B[i,j] = -1/A[k,k]
        else if(i == k & j != k)
          B[i,j] = A[i,j]/A[k,k]
        else if(i != k & j == k)
          B[i,j] = A[i,j]/A[k,k]
        else if(i != k & j != k)
          B[i,j] = A[i,j] - (A[i,k]*A[k,j]/A[k,k])
      }  
    }
    A = B
    #B = matrix(0, nrow = nrows, ncol = ncols)
  }  
  
  B = A
  ## The function outputs the matrix B
  return(B)
}


############################################################
## Function 2: Linear regression using the sweep operator ##
############################################################

myLinearRegression <- function(X, Y){
  
  # Find the regression coefficient estimates beta_hat
  # corresponding to the model Y = X * beta + epsilon
  # Your code must use the sweep operator you coded above.
  # Note: we do not know what beta is. We are only 
  # given a matrix X and a vector Y and we must come 
  # up with an estimate beta_hat.
  # 
  # X: an 'n row' by 'p column' matrix of input variables.
  # Y: an n-dimensional vector of responses

  #############################################
  ## FILL IN THE BODY OF THIS FUNCTION BELOW ##
  #############################################
  
    n <- nrow(X)
    p <- ncol(X)
    
    M = cbind(X,Y)
    N = t(M) %*% M 
    
    A = mySweep(N, p)
    beta_hat = A[1:p,p+1]
    
  ## Function returns the (p+1)-dimensional vector 
  ## beta_hat of regression coefficient estimates
  print(A[p+1,p+1])  #Sum of Squared errors (||e||)^2
  return(beta_hat)   #beta_hat should be p-dimensional for each of the p columns in X as said by Professor. Not sure why it says to return (p+1) dimensional vectors. Have emailed the same query. 
  
}

########################################################
## Optional examples (comment out before submitting!) ##
########################################################

testing_Linear_Regression <- function(){
  
  ## This function is not graded; you can use it to 
  ## test out the 'myLinearRegression' function 

  ## Define parameters
  # n    <- 100
  # p    <- 3
  
  # ## Simulate data from our assumed model.
  # ## We can assume that the true intercept is 0
  # X    <- matrix(rnorm(n * p), nrow = n)
  # beta <- matrix(1:p, nrow = p)
  # Y    <- X %*% beta + rnorm(n)
  
  # ## Save R's linear regression coefficients
  # R_coef  <- coef(lm(Y ~ X))
  
  # ## Save our linear regression coefficients
  # my_coef <- myLinearRegression(X, Y)
  
  # ## Are these two vectors different?
  # sum_square_diff <- sum((R_coef - my_coef)^2)
  # if(sum_square_diff <= 0.001){
  #   return('Both results are identical')
  # }else{
  #   return('There seems to be a problem...')
  # }
  
}

