#########################################################
## Stat 202A - Homework 5
## Author: Fuzail Mujahid Khan
## Date : 11/3/19
## Description: This script implements logistic regression
## using iterated reweighted least squares using the code 
## we have written for linear regression based on QR 
## decomposition
#########################################################

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
#############################################################

##################################
## Function 1: QR decomposition ##
##################################

myQR <- function(A){
  
  ## Perform QR decomposition on the matrix A
  ## Input: 
  ## A, an n x m matrix
  
  ########################
  ## FILL IN CODE BELOW ##
  ########################  

  n = dim(A)[1]
  m = dim(A)[2]
  
  require(Matrix)
  
  R = A 
  
  H_H = list() 
  
  for (k in 1:m) {
    x = R[k:n,k] 
    
    e = matrix(c(1, rep(0, length(x)-1)))
    
    V = sign(x[1]) * sqrt(sum(x^2)) * e + x
    
    hk = diag(length(x)) - 2 * as.vector(V %*% t(V)) / (t(V) %*% V)
    if (k > 1) {
      hk = bdiag(diag(k-1), hk)
    }
    
    H_H[[k]] = hk
    
    R = hk %*% R
  }
  
  Q = Reduce("%*%", H_H)    #Does the multiplication of householder matrices 
  
  ## Function should output a list with Q.transpose and R
  ## Q is an orthogonal n x n matrix
  ## R is an upper triangular n x m matrix
  ## Q and R satisfy the equation: A = Q %*% R
  
  return(list("Q" = t(Q), "R" = R))
  
}

###############################################
## Function 2: Linear regression based on QR ##
###############################################

myLinearRegression <- function(X, Y){
  
  ## Perform the linear regression of Y on X
  ## Input: 
  ## X is an n x p matrix of explanatory variables
  ## Y is an n dimensional vector of responses
  ## Do NOT simulate data in this function. n and p
  ## should be determined by X.
  ## Use myQR inside of this function
  
  ########################
  ## FILL IN CODE BELOW ##
  ########################  
  
  n = dim(X)[1]
  p = dim(X)[2]
  
  res = myQR(X)
  #print(res)
  
  out = res$Q %*% y 
  beta_hat = backsolve(res$R[1:p,1:p], out[1:p])
  
  resi = y - (X %*% beta_hat)
  error = sum(resi^2)
  
  
  ## Function returns the 1 x (p + 1) vector beta_ls, 
  ## the least squares solution vector
  return(list(beta_hat=beta_hat, error=error))
  
}

##################################################
## Function 3: Eigen decomposition based on QR  ##
##################################################
myEigen_QR <- function(A, numIter = 1000) {
  
  ## Perform PCA on matrix A using your QR function, myQR or Rcpp myQRC.
  ## Input:
  ## A: Square matrix
  ## numIter: Number of iterations
  
  ########################
  ## FILL IN CODE BELOW ##
  ######################## 
  
  ## Function should output a list with D and V
  ## D is a vector of eigenvalues of A
  ## V is the matrix of eigenvectors of A (in the 
  ## same order as the eigenvalues in D.)
  
  n = dim(A)[1] #X is square
  V = matrix(runif(n*n),n,n)
  A_c = A
  for (i in 1:numIter){
    res = myQR(V)
    Q = t(res$Q)
    V = A_c %*% Q
  }
  res = myQR(V)
  Q = t(res$Q)
  R = res$R
  
  return(list("D" = diag(R), "V" = Q))
}

###################################################
## Function 4: PCA based on Eigen decomposition  ##
###################################################
myPCA <- function(X) {
  
  ## Perform PCA on matrix A using your eigen decomposition.
  ## Input:
  ## X: Input Matrix with dimension n * p

  C = cov(X)
  res = myEigen_QR(C)
  Q = res$V
  Z = X %*% Q 
  
  ### They are equal
  #print(X)  
  #print(Z %*% t(Q))
  ###
  
  ## Output : 
  ## Q : basis matrix, p * p which is the basis system.
  ## Z : data matrix with dimension n * p based on the basis Q.
  ## It should match X = Z %*% Q.T. Please follow lecture notes.

  return(list("Q" = Q, "Z" = Z))
}
