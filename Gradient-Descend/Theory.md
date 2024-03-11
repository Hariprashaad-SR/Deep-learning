<h1><center>Using Gradient Descent Algorithm for Simple Linear Regression</center></h1>  
<center>By <b>HARIPRASHAAD SR</b></center>
<center>Student</center>

In machine learning, gradient descent is a popular optimization algorithm used to minimize the cost function of a model. In this tutorial, we'll explore how gradient descent can be applied to a simple linear regression model.

## Simple Linear Regression

Simple linear regression is a statistical method that allows us to summarize and study the relationships between two continuous variables. It assumes that there is a linear relationship between the independent variable (X) and the dependent variable (Y).

The equation of a simple linear regression model is given by:
`Y = mX + b + ε`


where:
- Y is the dependent variable
- X is the independent variable
- b is the intercept term
- m is the slope coefficient
- ε is the error term
## Loss Function in Simple Linear Regression

In simple linear regression, the loss function quantifies the difference between the predicted values of the dependent variable (Y) and the actual observed values. The goal of the regression model is to minimize this loss function, thereby finding the best-fitting line that describes the relationship between the independent variable (X) and the dependent variable (Y).

### Mean Squared Error (MSE)

The Mean Squared Error (MSE) is one commonly used loss function in simple linear regression. It is calculated as the average of the squared differences between the predicted values and the actual values:


$$ MSE = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2 $$ 



The MSE penalizes larger errors more heavily due to the squaring operation, making it sensitive to outliers.

## Gradient Descent

Gradient descent is an iterative optimization algorithm used to find the minimum of a function. In the context of machine learning, it is commonly used to minimize the cost function of a model by adjusting the model parameters (coefficients) iteratively.

The general steps of gradient descent are as follows:
1. Initialize the coefficients (parameters) of the model randomly or with some initial values.
2. Calculate the gradient of the cost function with respect to each coefficient.
3. Update the coefficients in the opposite direction of the gradient to minimize the cost function.
4. Repeat steps 2 and 3 until convergence or a specified number of iterations.

## Gradient Descent - Learning Rate

How big the steps gradient descent takes into the direction of the local minimum are determined by the learning rate, which figures out how fast or slow we will move towards the optimal weights.

For the gradient descent algorithm to reach the local minimum we must set the learning rate to an appropriate value, which is neither too low nor too high. This is important because if the steps it takes are too big, it may not reach the local minimum because it bounces back and forth between the convex function of gradient descent (see left image below). If we set the learning rate to a very small value, gradient descent will eventually reach the local minimum but that may take a while.

So, the learning rate should never be too high or too low for this reason. You can check if your learning rate is doing well by plotting it on a graph.

![image](https://i.ibb.co/JcxrpLy/0-v-DPz-Kbk0-IRE7iyd-T.jpg)
## Implementation Steps

### 1. Initialize Parameters
We start by initializing the intercept term (β0) and slope coefficient (β1) of the linear regression model.

### 2. Calculate Cost Function
We define a cost function (e.g., mean squared error) to measure the difference between the predicted values and the actual values of the dependent variable.

### 3. Update Parameters
Using gradient descent, we iteratively update the parameters (coefficients) of the model to minimize the cost function.

### 4. Repeat Until Convergence
We repeat steps 2 and 3 until the cost function converges to a minimum value or until a specified number of iterations is reached.

## Conclusion

Gradient descent is a powerful optimization algorithm that is widely used in machine learning for training models. In the context of simple linear regression, it allows us to find the optimal values of the intercept and slope coefficients by minimizing the cost function.

<!-- Replace 'image1.jpg', 'image2.jpg', and 'image3.jpg' with the actual filenames or URLs of your images -->
<img src="https://i.ibb.co/2dnSt5d/Screenshots-2024-03-11-at-11-03-59-PM.png" alt="Image 1" style="float:left; margin-right:10px;" width="300"/>
<img src="https://i.ibb.co/V3VnSg2/Screenshots-2024-03-11-at-11-04-17-PM.png" alt="Image 2" style="float:left; margin-right:10px;" width="300"/>
<img src="https://i.ibb.co/VgPSzbf/Screenshots-2024-03-11-at-11-18-30-PM.png" alt="Image 3" style="float:left; margin-right:10px;" width="300"/>
<img src="https://i.ibb.co/MR9F5rR/Screenshots-2024-03-11-at-11-04-30-PM.png" alt="Image 3" style="float:left; margin-right:10px;" width="300"/>


*Figure 1: Visualization of Gradient Descent Algorithm*
