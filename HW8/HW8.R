Prepare_data <- function()
{
 
  df = read.csv('../Downloads/mnist_test.csv', header = FALSE, sep = ",")
  df = df[df$V1 == 0 | df$V1 == 1,]
  
  m = dim(df)[1]
  n = dim(df)[2]
  
  X = df[,2:n]
  Y = df[,1]
  
  for (i in 1:m) { 
    if(Y[i] == 0)
      Y[i] = -1
  }
  
  train_size = 1000
  X_train = X[1:train_size,]
  Y_train = Y[1:train_size]
  print(dim(X_train))
  print(dim(Y_train))
  
  X_test = X[(train_size+1):m,]
  Y_test = Y[(train_size+1):m]
  print(dim(X_test))
  print(dim(Y_test))
  
return(list("X_train" = X_train, "Y_train" = Y_train, "X_test" = X_test, "Y_test" = Y_test))
  
}  
  

mySVM <- function(X_train, Y_train, X_test, Y_test, kernel='rbf', training_method='grad_desc')
{
  
X = X_train
Y = Y_train

print(X) 
print(Y)
print(dim(X))
print(dim(Y))

m = dim(X)[1]
F = matrix(0L, nrow = m, ncol = m)

### This is RBF kernel
if(kernel == 'rbf') { 
  print('Performing RBF Kernel calculations')
  for (i in 1:m){
    for (j in i:m){
      print(paste(i,j))
      F_val = exp((-0.5)*(norm(as.matrix((X[i,])-X[j,]))**2))
      F[i,j] = F_val
      F[j,i] = F_val
    }
  }
}

### This is linear kernel
#if(kernel == 'linear') {
#for (i in 1:m){
#    for (j in i:m){
#    print(paste(i,j))
#    F[i,j] = (X[i] %*% X[j])
#    F[i,j] = F_val
#    F[j,i] = F_val
#}
#}
#}

### This is Polynomial kernel
#if(kernel == 'polynomial') {
#gamma = 1
#c = 1
#for (i in 1:m){
#    for (j in i:m){
#    print(paste(i,j))
#    F[i,j] = (X[i] %*% X[j])**2
#    F[i,j] = F_val
#    F[j,i] = F_val
#}
#}
#}

#This is sigmoid kernel
#if(kernel = 'sigmoid') {
#for (i in 1:m):
#    for (j in i:m):
#        print(paste(i,j))
#        F[i,j] = tanh(10*(X[i] %*% X[j]) + 5)
#         F[i,j] = F_val
#         F[j,i] = F_val
#}
#}
#}

print(dim(F))

w = matrix(0L, nrow = m, ncol=1)
eta = 0.05
epochs = 10

iteration_list = numeric()
accuracy_list = numeric()

for (epoch in 1:epochs){
  for (i in 1:m){
  if ((Y[i]*(F[i,] %*% w)) < 1)
  w = w + eta * ( (F[i,] * Y[i]) + (-2  *(1/epoch)* w) )
  else
  w = w + eta * (-2  *(1/epoch)* w)
  
  y_pred = F %*% w
  for (i in 1:m){
  if(y_pred[i] > 0)
  y_pred[i] = 1
  else
  y_pred[i] = -1
  }
  truth = as.numeric(Y == y_pred)
  print(mean(truth))
  iteration_list = append(iteration_list,(((epoch-1)*m)+i))
  accuracy_list = append(accuracy_list, mean(truth))
  }
}

print(iteration_list)
print(accuracy_list)

}