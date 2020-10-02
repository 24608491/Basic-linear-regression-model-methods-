############################################################################
options(max.print=1000000)
trainX=read.csv("trainX.csv",header = F,sep = ",")
x=as.matrix.data.frame(trainX)
trainy=read.csv("trainy.csv",header = F,sep = ",")
y=as.matrix.data.frame(trainy)
                   
##############################################################################
#create a vector of ones for theta (coeffecient vector)
x0 <- rep(1, nrow(x))
x <- data.frame(cbind(x0,x))

# merge x and y to enable accurate random sampling
xy <- data.frame(cbind(y,x))
#STOCHASTIC GRADIENT DESCENT
################################################################################
# define a function that implements the partial derivative component of 
# the gradient descent formula
g_part_deriv <- function(x, y, theta) {
  part_deriv <- (1/ nrow(y))* (t(x) %*% ((x %*% t(theta)) - y))
  return(t(part_deriv))
}


##################################################################################
#stochoastic gradient descent (sampling 1 data point for each iteration)
# define stochastic gradient descent algorithm
#x is the 
stoch_gradDescent <- function(x, y, alpha, n){
  # theta MUST be 2 columns: must match width of x matrix
  # set matrix theta and set all elements = 0 to start with
  theta <- matrix(c(0, 0), nrow = 1)
  # Initialize a matrix to store values of theta for each iteration
  thetaIter <- matrix(NA, nrow = n, ncol = 2)
  # set seed value for random sampling
  set.seed(42)
  #loss function/mean squared error initialization
   loss=c()
  # now iterate using stochastic gradient (randomly sampled  data point), updating theta each step
  for (i in 1:n) {
    # randomly sample 1 item from the combined xy data frame
    xysamp <- as.matrix( xy[sample(nrow(xy), 1, replace = TRUE), ] )
    # isolate 'x' component of random samples
    xsamp <- as.matrix(xysamp[,2:3])
    # isolate 'y' component of random samples
    ysamp <- as.matrix(xysamp[,1])
    # update theta using stochastic gradient
    theta <- theta - alpha  * g_part_deriv(t(xsamp), ysamp, theta) 
    #calculate the loss
    loss= c(loss,(1/ nrow(ysamp))*(sum((t(ysamp) - theta%*%xsamp)^2)))
  } # end for loop
  return(list(theta,loss)) 
}
#####################################################################
results=stoch_gradDescent(x=x,y=y,alpha=0.01,n=300)
#obtaining the coefficients from the final iterations 
parame=results[[1]]
los=results[[2]]
epo=seq(1:300)
plot(epo,los,cex=0.5,xlab='# of epochs',ylab='mean squared error')
lines(epo,los)

#####################################################################
#batchsize of 10 was used during iteration
minibatch_gradDescent <- function(x, y, alpha, n){
  theta <- matrix(c(0, 0), nrow = 1)
  thetaIter <- matrix(NA, nrow = n, ncol = 2)
  set.seed(42)
  loss=c()
  for (i in 1:n) {
    # randomly sample 10 items from the combined xy data frame
    xysamp <- as.matrix( xy[sample(nrow(xy), 10, replace = TRUE), ] )
    xsamp <- as.matrix(xysamp[,2:3])
    ysamp <- as.matrix(xysamp[,1])
    theta <- theta - alpha  * g_part_deriv(xsamp, ysamp, theta)
    thetaIter[i,] <- theta
    loss = c(loss,(1/ nrow(ysamp))*(sum((t(ysamp) - theta%*%t(xsamp))^2)))
  } 
  return(list(theta,loss))
}

results2=minibatch_gradDescent(x=x,y=y,alpha=0.01,n=300)

parame2=results2[[1]]
los2=results2[[2]]
epo2=seq(1:300)
plot(epo2,los2,cex=0.5,xlab='# of epochs',ylab='mean squared error')
lines(epo2,los2)


########################################################################
#using a multi perceptron network
#
trainX=read.csv("trainX.csv",header = F,sep = ",")
xx=as.matrix.data.frame(trainX)
trainy=read.csv("trainy.csv",header = F,sep = ",")
yy=as.matrix.data.frame(trainy)
####
#

library(tensorflow)
library(keras)

#1 creating the multilayer perceptron
#2 created with one hidden layer that had 5 neurons
#3 relu function was chosen as activation function
#in hidden layer since it allows backpropagation
#4 activation function for output layer was linear since the neural network
#is modeling a linear relationship

model<- keras_model_sequential()
model %>%
  layer_dense(units=5,activation='relu',input_shape = c(1)) %>%
  layer_dense(units=1, activation='linear')

summary(model)

#compile 
#ADAM optimizer was used during optimization
model %>% compile(
  optimizer = 'adam',
  loss = 'mse',           # mse is the mean squared error
  metrics = list('mse') )

#Fit/train the model
model %>% fit(xx,yy,epochs = 50,batch_size = 10)


####################################################################
#test data
testX=read.csv("testX.csv",header = F,sep = ",")
tesx=as.matrix.data.frame(testX)
testy=read.csv("testy.csv",header = F,sep = ",")
tesy=as.matrix.data.frame(testy)
#used error between predicted and actual to check performance
#predicted y values from the 3 methods with the actual y values
#########################################################################
#creating prediction model from parameters found using SGD
m=parame[2]
c=parame[1]

#prediction model to predict y 
y1=tesx*m+c

#calculate Root mean squared error
err1=y1-tesy
err1=as.matrix.data.frame(err1)
#plot error distribution
hist(err1)
plot(y1,tesy,xlab='predicted y values',ylab='actual y values',cex=0.6)
rmse1=sqrt(mean(err1^2))

#creating prediction model from parameters found using mini-batch
m2=parame2[2]
c2=parame2[1]

#prediction model to predict y 
y2=tesx*m2+c2

#calculate Root mean squared error
err2=y2-tesy
err2=as.matrix.data.frame(err2)
#plot error distribution
hist(err2)
plot(y2,tesy,xlab='predicted y values',ylab='actual y values',cex=0.6)

rmse2=sqrt(mean(err2^2))
########################################################################
#prediction from MLP
score <- model %>% evaluate(tesx,tesy)
y3<- model %>% predict(tesx)

err3=y3-tesy
err3=as.matrix.data.frame(err3)

hist(err3)
plot(y3,tesy,xlab='predicted y values',ylab='actual y values',cex=0.6)
rmse3=sqrt(mean(err3^2))
