import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_cost(n, y, y_pred):
    # Calculating the cost function
    cost = (1/n) * (sum([i ** 2 for i in (y - y_pred)]))
    return cost

def partial_diff_m(n, y, y_pred, x):
    # Partial differentiation of the loss function wrt m
    md = -(2/n) * sum((y - y_pred) * (x))
    return md

def partial_diff_b(n, y, y_pred):
    # Partial differentiation of the loss function wrt b
    bd = -(2/n) * sum((y - y_pred))
    return bd


def gradient_descend(x, y):
    # Function to implement the gradient descend algorithm for linear regression
    m_curr, b_curr = 0, 0
    learning_rate = 0.001
    epochs = 10000
    n = len(x)
    prev_cost = 100

    for epoch in range(epochs):
        y_pred = (m_curr * x) + b_curr
        cost = calculate_cost(n, y, y_pred)
        if abs(prev_cost - cost) < 10:
            # If cost difference is less than *(10) abort the training
            break
        
        plt.clf()
        plt.scatter(x, y)
        plt.title(f'epochs : {epoch} cost : {round(abs(prev_cost - cost), 2)}')
        plt.plot(x, y_pred)
        plt.pause(0.001)

        md = partial_diff_m(n, y, y_pred, x)
        bd = partial_diff_b(n, y, y_pred)

        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd

        prev_cost = cost
    
    return (m_curr, b_curr)


if __name__ == '__main__': 
    df = pd.read_csv('Assets/Salary_Data.csv')
    
    x = np.array(df.iloc[: , 0])
    y = np.array(df.iloc[:, 1])

    m, b = gradient_descend(x, y)
    print(f'Slope(m) : {m} and bias(b) : {b}')