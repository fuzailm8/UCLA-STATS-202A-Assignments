#########################################################
## Stat 202A - Homework 6
## Author: Fuzail Mujahid Khan  
## Date : 11/11/19
## Description: See CCLE
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

myLM <- function(X, Y){
  
  ## Perform the linear regression of Y on X
  ## Input: 
  ## X is an n x p matrix of explanatory variables
  ## Y is an n dimensional vector of responses
  ## Use myQR inside of this function
  
  ########################
  ## FILL IN CODE BELOW ##
  ########################  
  
  n = dim(X)[1]
  p = dim(X)[2]
  
  res = myQR(X)
  #print(res)
  
  #print(dim(res$Q))
  #print(dim(y))
    
  out = res$Q %*% Y 
  beta_ls = backsolve(res$R[1:p,1:p], out[1:p])
  beta_ls = matrix(beta_ls, nrow = p, ncol = 1)
  
  ## Function returns the 1 x p vector beta_ls, notice this version do not add intercept.
  ## the least squares solution vector
  return(beta_ls)
  
}

######################################
## Function 3: Logistic regression  ##
######################################

## Expit/sigmoid function
expit <- function(x){
  1 / (1 + exp(-x))
}

myLogisticSolution <- function(X, Y){

  ########################
  ## FILL IN CODE BELOW ##
  ########################
  
  cost_func <- function(beta, X, Y){
    
    n = dim(X)[1]
    
    Y_hat = expit(X %*% beta)
    J = (-1/n) * ((t(Y) %*% log(Y_hat) + t(1-Y) %*% log(1-Y_hat)))
    J
  }
  
  gradient <- function(beta, X, Y){
    
    n = dim(X)[1] 
    Y_hat = expit(X %*% beta)
    
    grad = (1/n) * (t(X) %*% (Y_hat - Y)) 
    grad
  }
  
  lr = 0.01
  epsilon = 0.00000001 
  
  beta = myLM(X,Y)
  print(beta)
  
  cost = cost_func(beta, X, Y) 
  error = 1
  
  print(error) 
  print(epsilon)
  
  while (error > epsilon){ 
  cost_prev = cost 
  beta = beta - (lr * gradient(beta, X, Y)) 
  cost = cost_func(beta, X, Y) 
  error = cost_prev - cost 
  #print(error)
  }
  
  Y_hat = expit(X %*% beta)
  print(Y_hat)
  
  return(beta)
}

###################################################
## Function 4: Adaboost  ##
###################################################

  
  training <- function(X, w, y) {
    
    p = ncol(X)
    
    min_error <- c()
    min_cut <- c()
    min_m <- c()
    
    for (j in 1:p) {
      x_j = X[,j]
      err_cuts = c()
      cuts = unique(x_j)
      y_pred = rep(NA, length(x_j))
      
      m_list = c()
      
      for (t in 1:length(cuts)) {
        y_pred[x_j > cuts[t]] = 1
        y_pred[x_j <= cuts[t]] = -1
        
        m = ifelse(mean(y_pred == y) > 0.5, 1, -1)
        err_cuts[t] = sum(w*(y != y_pred*m))/sum(w)
        m_list[t] = m
      }
      
      min_idx = which.min(err_cuts)
      min_error[j] = min(err_cuts)
      min_cut[j] = cuts[min_idx]
      min_m[j] = m_list[which.min(m_list)]
    }
    
    opt_feature = which.min(min_error)
    cut_point = min_cut[opt_feature]
    m = min_m[opt_feature]
    
    return(c(opt_feature, cut_point, m))
  }
  
  classify <- function(X, pars) {
    opt_feature = pars[1]
    cut_point = pars[2]
    m = pars[3]
    
    x_j = X[,opt_feature]
    pred = rep(NA, nrow(X))
    pred[x_j > cut_point] = m
    pred[x_j <= cut_point] = -m
    
    return(pred)
  }
  
  vote <- function(X, alpha, dec_tree_par) {
    alpha = matrix(alpha)
    y_hat_total = sapply(dec_tree_par,function(pars) classify(X, pars))
    y_hat = y_hat_total %*% alpha
    
    y_hat[y_hat>=0] = 1
    y_hat[y_hat<0] = -1
    
    return(y_hat)
    
  }
  
  engine_adaboost <- function(X, y, B) {
    e = NA
    dec_tree_par = list()
    alpha = c()
    n = nrow(X)
    
    w = rep(1/n, n)
    
    for(b in 1:B) {
      par = training(X, w, y)
      y_hat = classify(X, par)
      
      dec_tree_par[[b]] = par
      e = sum(w*(y != y_hat))/sum(w)
      
      alpha_b = log((1-e)/e)
      alpha[[b]] = alpha_b
      w <- w*exp(alpha_b*(y != y_hat))
    }
    
    print(paste('Error rate e = ',e))
    print(paste('Parameter list for dec tree = ',dec_tree_par))
    print(paste('Parameter list Alpha = ',alpha))
    
    return(list(alpha = alpha, dec_tree_par = dec_tree_par))
  }
  
  start_adaboost <- function(train, test, B) {
    
    X_train = train[,-1]
    y_train = train[,1]
    
    X_test = test[,-1]
    y_test = test[, 1]
    
    test_err = matrix(NA, nrow = 1, ncol = B)
    train_err = matrix(NA, nrow = 1, ncol = B)
    
    for (b in 1:B) {
      parameters = engine_adaboost(X_train, y_train, b)
      
      train_pred = vote(X_train, parameters$alpha, parameters$dec_tree_par)
      train_err[b] = mean(train_pred != y_train)
      
      test_pred = vote(X_test, parameters$alpha, parameters$dec_tree_par)
      test_err[b] = mean(test_pred != y_test)
    }
    
    print(train_err)
    print(test_err)
    
    return(list(train_err, test_err, train_pred, test_pred))
  }
  
  
  myAdaboost <- function(x1, x2, y) {
    
    library(reshape2)
    library(ggplot2)
    
    data = cbind(y,x1,x2)
    
    n = nrow(data)
    data = as.data.frame(data) 
    
    for (i in 1:n){
      if(data$y[i] == 0)
        data$y[i] = -1
    }
    
    train = data[1:(0.8*n),]
    test = data[((0.8*n)+1):n, ]
    
    B = 10 #No. of decision trees
    
    res = start_adaboost(train, test, B)
    
    train_pred = res[[3]]
    test_pred = res[[4]]
    
    train_pred = as.factor(train_pred)
    test_pred = as.factor(test_pred)
    
    plot(train$x1, train$x2, col = c('red','blue')[train_pred])
    plot(test$x1, test$x2, col = c('red','blue')[test_pred])
    
    
    
    train_error = res[[1]]
    test_error = res[[2]]
    
    train_error_mean = apply(train_error, 2, mean)
    test_error_mean = apply(test_error, 2, mean)
    
    results = data.frame(Train = train_error_mean, Test = test_error_mean)
    
    results_melt = melt(results)
    results_melt$B = rep(1:B, 2)
    colnames(results_melt) = c("Error Type", "Error Rate", "B - No. of decision trees")
    ggplot(results_melt) + geom_line(aes(`B - No. of decision trees`,`Error Rate`, colour = `Error Type`))
    
    
  }
  

###################################################
## Function 5: XGBoost  ##
###################################################

myXGBoost <- function(x1, x2, y) {

}

## Simulation

test <- function() {

  # Test (1)
  n <- 5000
  p <- 4
  
  X    <- matrix(rnorm(n * p), nrow = n)
  beta <- c(12, -2,-3, 4)
  Y    <- 1 * (runif(n) < expit(X %*% beta))
  
  ## Our solution
  logistic_beta <- myLogistic(X, Y)
  logistic_beta    
  
  ## R's solution
  coef(glm(Y ~ X + 0, family = binomial(link = 'logit')))


  # Test (2, 3)

  num_sample <- 10000

  x1 <- runif(num_sample)
  x2 <- runif(num_sample)
  y <- as.integer((x1^2+x2^2 < 1))

  myAdaboost(x1, x2, y)
  myXGBoost(x1, x2, y)
  
}
