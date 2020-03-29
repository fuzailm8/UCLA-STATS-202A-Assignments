# UCLA-STATS-202A-Assignments
Course assignments from UCLA Statistics course - Statistical Programming (STATS202A) (Fall 2019) 

This report is intended to give a brief overview about the assignments implemented in the STATS202A course. There were 8 HW assignments and below are comments about each of the HW assignments. 

HW 1 - Random Number Generation and Monte Carlo Sampling 
HW 2 - Metropolis-Hastings and Gibbs Sampling
HW 3 - Linear Regression using the Sweep Operation 
HW 4 - QR Decomposition and Linear Regression 
HW 5 - Eigen decomposition and PCA
HW 6 - Logistic Regression and Adaboost
HW 7 - XGBoost using Python Packages 
HW 8 - Support Vector Machines 

## HW 1 - Random Number Generation and Monte Carlo Sampling
Languages used: Python and R 
The below random number generators were coded: 
- Uniform distribution [0,1] using the Linear Congruential Generator (LCG) algorithm. It is defined by the formula:
![{\displaystyle X_{n+1}=\left(aX_{n}+c\right){\bmod {m}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/3a40cd0032b03626a091a5a0e1b4684b3d5eb406)
X~n~ and X~n+1~ are terms in the random number sequence. The 'm' is the modulus (m>0), 'a' is the multiplier (0<a<m) and 'c' is the increment (0<=c<m). 

	***Function Inputs*** : Lower limit, Upper limit. 
	***Function Outputs***: None
- Exponential (1) distribution using the inversion method:
Given a sequence of uniform random numbers (u), we can generate an exponential distribution from this uniform random number sequence using the formula:  
***y = -log(1-u)/k*** where u is a uniform random number and k is the parameter for e<sup>k</sup> which is default 1 in our case. 

  ***Function Inputs***: k  
  ***Function Outputs***: None
- Normal (0,1) using the polar method: 
The aim is to create a normal distribution with a specified mean (default 0) and variance (default 1) in our case. We convert two sequences X and Y of uniform random numbers into Polar Coordinates (R, &theta;). Then using the formulas: ***X = Rcos&theta;*** and ***Y = Rsin&theta;*** , we obtain our normal distribution. 

	***Function Inputs***: Mean, Variance. 
	***Function Outputs***: None


Monte Carlo Sampling: 
- Computation of Pi: We generate uniform random numbers in X and Y to populate the unit square [0,1]. We then calculate the frequency of points that fall below ***x<sup>2</sup> + y<sup>2</sup> = 1***. Using the area formulas of square and circle, we can conclude that: 
***pi = (no. of points that fall below equation/ total. no of points) * 4*** 
This is obtained from the area formulas of circle and square. 
- Estimate volume of 5-d and 10-d unit ball: This is just taking the similar concept explained above into higher dimensions. 
For d = 5, 
***Vol of ball = (no. of points in unit ball/no. of points in unit cube) * 2<sup>5</sup>***
For d = 10, 
***Vol of ball = (no. of points in unit ball/no. of points in unit cube) * 2<sup>10</sup>***


***Function Inputs*** = d
***Function Outputs*** = None
  
## HW2 - Metropolis-Hastings and Gibbs Sampling
### Part 1 - Sampling using Metropolis-Hastings algorithm
A) ***Function Name***: sample_uniform
	***Function Inputs*** = size, lower limit, upper limit 
	***Function Outputs*** = Uniform sequence 
	
We create a uniform random number sequence like we have done previously based on function inputs received. 
	
B) ***Function Name***: sample_normal_chain
	***Function Inputs*** = x0, c, chain length, mean, variance 
	***Function Outputs*** = 2D matrix with dimension: (no. of chains, chain length)
	
x0 is a uniform number initial sequence of length = 1000.
 Given mean and variance, this function would sample from normal distribution N (mean, variance)  creating multiple chains (1000) of chain length (100). The proposal distribution at x is ***y ~ U[x-c, x+c]*** . 
 
C) ***Function Name***: metropolis_simulation
***Function Inputs*** = num_chain, chain_length, mean, variance
***Function Outputs*** = Output GIF

Here we create different settings for Metropolis sampling, intitialize x0 and call sample_normal_chain with these configurations. 
With our 1000 chains of length 100 each, we create an output GIF to show the change of histogram of 1000 points each across 100 timesteps. 

### Part 2 - Gibbs Sampling 
A) ***Function Name***: gibbs_sample 
***Function Inputs***: x0, y0, rho, num_chain, chain_length, mean, variance 
***Function Outputs***: Matrix of dimensions (num_chain, chain_length,2)

Using the loops: 
-> ***x[i] = N * $\sqrt{1-rho^2}$ + rho$*$y[i-1]*** 
-> ***y[i] = N * $\sqrt{1-rho^2}$ + rho$*$x[i]*** 

The above formula forms the basis for Gibbs sampling where we sample from a bi-variate normal distribution N with correlation coefficient rho. 
We return the output matrix of dimension (1000,100,2) from this function. 

B) ***Function Name***: gibbs_simulation
***Function Inputs***: None
***Function Outputs***: Scatterplot movie and movie of chain footsteps 

This function is for calling gibbs_sample with different values of rho and using the returned matrix to output movie of histogram and output movie of chain footsteps. 

## HW3 - Sweep Operation

A) ***Function Name***: mySweep
***Function Inputs***: A, k
A - a square matrix (k,k) 
k - to specify pivot A[k,k]
***Function Outputs***: B
B - Swept matrix (k,k) 

Using the Sweep operation, we create the swept matrix sweeping through 1:k  

B) ***Function Name***: myLinearRegression
***Function Inputs***: X, Y 
X - an (n,p) input matrix with n samples and p features
Y - an (n,1) response matrix 
***Function Outputs*** - beta_hat 
beta_hat - (p+1) dimensional vector of linear regression coefficient estimates

Given input X and response Y, we use this function to use the Sweep operator to obtain the linear regression coefficient estimates, beta. Using the formula: 
***Y = X$*$Beta + Intercept***
X - (n,p) 
B - (p,1) 
Intercept - (n,1)
Y - (n,1) 
Derived from the concept of normal equations and differentiation, the mathematical formula for estimate of beta from X and Y is:
 
**Beta Estimate = (X<sup>t</sup>X)<sup>-1</sup>X<sup>t</sup>Y**

## HW4 - QR Decomposition and Linear Regression
A) ***Function Name***: myQR
***Function Inputs***: A 
A - n x m matrix. QR decomposition of this matrix, A = QR  
***Function Outputs***: Q<sup>t</sup> and R 
Q - orthogonal n x n matrix 
R - upper triangular n x m matrix  

Using householder transformations, the input matrix A is decomposed as the product of an orthogonal matrix Q and an upper triangular matrix R.  

B) ***Function Name***: myLinearRegression 
***Function Inputs***: X, Y
X - input n x p matrix 
Y - output n x 1 matrix 
***Function Outputs***: Beta estimate and least squares solution vector 

We call myQR on the matrix X and perform QR decomposition on the matrix X. From this, we obtain beta_hat and sum of squared errors is the sum of the squared errors between the actual y and the predicted y. 

## HW 5 - Eigen decomposition and PCA

A) ***Function Name***: myQR
***Function Inputs***: A 
A - n x m matrix. QR decomposition of this matrix, A = QR  
***Function Outputs***: Q<sup>t</sup> and R 
Q - orthogonal n x n matrix 
R - upper triangular n x m matrix  

Using householder transformations, the input matrix A is decomposed as the product of an orthogonal matrix Q and an upper triangular matrix R.  

B) ***Function Name***: myLinearRegression 
***Function Inputs***: X, Y
X - input n x p matrix 
Y - output n x 1 matrix 
***Function Outputs***: Beta estimate and least squares solution vector 

We call myQR on the matrix X and perform QR decomposition on the matrix X. From this, we obtain beta_hat and sum of squared errors is the sum of the squared errors between the actual y and the predicted y. 

C) ***Function Name***: myEigen_QR
***Function Inputs***: A, numIter
A - square matrix
numIter - number of iterations 
***Function Outputs***: D, V 
D - Vector of eigenvalues of A 
V - Matrix of eigenvectors of A 

By iteratively performing QR decomposition calling myQR for numIter iterations, we obtain the eigenvalues D and the eigenvectors V of input matrix A. If A is n x n, we have n eigen values and n x n eigen vector matrix. 

D) ***Function Name***: myPCA
***Function Inputs***: X (input matrix n x p) 
***Function Outputs***: Q, Z 
Q - new basis matrix, p x p
Z - new data matrix, n x p  

We calculate the covariance matrix of input matrix X and pass this as input to myEigen_QR. 
After performing PCA, the new basis vectors for X are the eigen vectors of the covariance matrix of X which we call Q. To obtain the new data points, we use equation ***Z = X $*$ Q***. 
Therefore, we now have the new basis vectors after PCA (Q) and new data points matrix (Z).  

## HW6 - Logistic Regression, Adaboost and XGBoost

A) ***Function Name***: myQR
***Function Inputs***: A 
A - n x m matrix. QR decomposition of this matrix, A = QR  
***Function Outputs***: Q<sup>t</sup> and R 
Q - orthogonal n x n matrix 
R - upper triangular n x m matrix  

Using householder transformations, the input matrix A is decomposed as the product of an orthogonal matrix Q and an upper triangular matrix R.  

B) ***Function Name***: myLM 
***Function Inputs***: X, Y
X - input n x p matrix 
Y - output n x 1 matrix 
***Function Outputs***: Beta estimate (1 x p)  

We call myQR on the matrix X and perform QR decomposition on the matrix X. From this, we obtain beta_hat. 

C) ***Function Name***: myLogisticSolution
***Function Inputs***: X, Y 
***Function Outputs***: beta 

We pass the X and Y to the myLM function to get the linear regression coefficient estimates beta. We then use the sigmoid function:
***g(z) = 1/(1+e<sup>-z</sup>)*** 
where z = X $*$ B 

Here, we use the cost function: 
***J = (-1/n) * (ylog(y_hat) + (1-y)log(1-y_hat))*** 
where n = no. of training samples 
y_hat = predicted y   

We then apply gradient descent to update the coefficients B: 
***B = B - (learning_rate * (dJ/dB))*** 

After this process, we get our output  beta estimate.
 
D) ***Function Name***: myAdaboost 
***Function Inputs***: x1,x2,y 
x1, x2 - two uniform random sequences 
y - 0/1 label to indicate if ***x1<sup>2</sup> + x2<sup>2</sup> < 1***
***Function Outputs***: None 

   The key overall processes involved here are: 
   ***Splitting*** - Split the data into a training and testing data set. 
   ***Training*** - Looping through each feature and trying to find the optimal feature and optimal cut point at which decision boundary can be made. 
   ***Classifying*** - Making predictions on our input based on these decision boundaries for each decision tree. 
   ***Voting*** - Using the significance of each decision tree, perform the voting process to classify an input as a label based on the decision boundaries made.  
***Boosting*** - Looping through the forest and building the trees iteratively and hence keep updating weights through each iteration and update significance &alpha; of each tree. This is to perform the boosting process.  
***Tracking accuracy***- Calculate the training error and testing error per iteration. 

## HW7 - XGBoost using Python packages

***Overview*** - This was an exercise to perform XGBoost in practise using packages in Python such as xgboost, sklearn, numpy and matplotlib. We worked on the in-built cancer data set to perform a supervised classification problem. 

***Exercise 1*** - We perform 5 fold cross validation on the data and obtain the mean and standard deviation of the 5 fold cross validation accuracy. This is to get a better picture of how the XGB classifier is performing. 

***Exercise 2*** - We then performed Grid Search for hyper parameter tuning to get the most optimal parameters. We specified ranges in values for variables max_depth and min_child_weight.
We then calculated the mean test scores for these different configurations. For each configuration, 5 fold CV was performed here as well. 

***Exercise 3*** - After performing hyper parameter tuning and cross fold validation, we find the optimal hyper parameters and create the best model. After this, we plot the feature importance of that model. 
  
## HW8 - Support Vector Machines (SVM)
A) ***Function Name***: Prepare_data
***Function Inputs***: None 
***Function Outputs***: Train data, train labels, test data, test labels 

In this function, we load the MNIST data set and partition the data into training set and testing set. Since we are only classifying on 2 classes - 0 and 1, we only take the relevant data that corresponds to these 2 classes. 

B) ***Function Name***: mySVM
***Function Inputs***: train_data, test_data, train_label, test_label, kernel, training method 
***Function Output***: None

Depending on the kernel type specified (gaussian, linear, polynomial, sigmoid), we create the high dimensional feature matrix with the kernel features in it to transform the present input features to higher dimensional features using the kernel method.
 
| Kernel  | 	Equation |
|--|--|
| Gaussian | ***e<sup>(-gamma$*$(u-v)<sup>2</sup>)</sup>***  |
|Linear|***u' $*$ v***|
| Polynomial | ***(gamma$*$u'$*$v + c)<sup>2</sup>*** |
| Sigmoid | ***tanh(gamma$*$u'$*$v + c)*** |

 

Now further depending on the training method specified as '***Gradient descent***' or '***Dual coordinate ascent***', that particular method is applied. Both of the methods will keep training until the optimal weights are calculated. As we update parameters, we keep track of the loss at each iteration to finally plot the accuracy per iteration graph. 

C) ***Function Name***: test
***Function Inputs***: train_data, test_data, train_label, test_label, kernel, training method
***Function Output***: None 

In this function, we try different settings, different kernel types, different training methods and obtain results and plot the accuracy result and accuracy per iteration for these different configurations.
 
For example, for the Gaussian kernel, you can experiment with different values of gamma and see the effect on training and accuracy after we plot the different settings on one graph. 
