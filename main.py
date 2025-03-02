import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

N=100

'''
    Equation y = m1 * x + m2 + Gaussuian Noise(0,1)
'''

# True m1 and m2 values
m1,m2=6.9,2.5
x=np.linspace(-10,10,N)

# Gaussuian Noise value
noise=np.random.normal(0,1,N)

# Value for y using the above mentioned equation
y=m1 * x + m2 + noise

# Function to calculate Mean Squared Error(MSE) 
def mse(y_pred, y):
    return np.mean((y - y_pred)**2)

'''
    y_pred = m * x
    MSE = (1/n) * ∑(y- y_pred)^2
'''

m_values = np.linspace(-10, 10, N)
losses = [mse(m * x ,y) for m in m_values]

# Linear search to find best value for 'm' with minimum loss
best_m = 0
min_loss = float('inf')

for m, loss in zip(m_values,losses):
    if loss < min_loss:
        best_m = m
        min_loss = loss
print(f"Best value for 'm' through Linear Search: {best_m}")


plt.figure(figsize=(15, 10))

# Plot graph for Linear Search result
def plot_LinearSearch():
    plt.plot(m_values, losses, label='MSE')
    plt.axvline(best_m, color='red', linestyle="--", label=f'Best "m": {best_m:.2f}')
    plt.xlabel('m values')
    plt.ylabel('MSE')
    plt.title('Linear Search "m" vs MSE')
    plt.legend()
    plt.show()

plot_LinearSearch()

'''
    To find the gradient of the loss function we have to do the following,
        partial differentiation wrt m
        d(MSE)/dm = 1/n * 2 * ∑[ (y-m * x) * -x ]
                  = -2 * Mean of [ x * (y - y_pred) ]
'''

# Gradient Descent to find the best value for 'm' with minimum loss
epochs = 100
learning_rate = 0.01

# Starting 'm' value for Gradient Descent
m_gd = np.random.randn()

# To store the 'm' values over each iterations
m_gd_values = []

# To store the loss values over each iterations
loss_values = []

for _ in range(epochs):
    loss_values.append(mse(m_gd * x, y))
    # From above mentioned formula
    gradient = -2 * np.mean(x * (y - (m_gd * x)))
    m_gd_values.append(m_gd)
    m_gd -= learning_rate * gradient

print(f"Best value for 'm' through Gradient Descent: {m_gd}")

# Plot graph for Gradient Descent
def plot_GradientDescent():
    # Finding the ranges for 'm' and loss to plot graph
    m_gd_range = np.linspace(min(m_gd_values)-1, max(m_gd_values)+1, 100)
    loss_range = [mse(m1 * x, y) for m1 in m_gd_range]

    plt.plot(m_gd_range, loss_range, label='MSE', color='blue')
    plt.scatter(m_gd_values, loss_values, color='red', marker='o', label='Gradient Descent iterations')

    pts = 10
    ind = np.linspace(0,len(m_gd_values)-1,pts,dtype=int)

    for i in ind:
        plt.annotate(f"{m_gd_values[i]:.2f}", (m_gd_values[i], loss_values[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.xlabel('m values')
    plt.ylabel('MSE')
    plt.title('Gradient Descent "m" vs MSE')
    plt.legend()
    plt.show()

plot_GradientDescent()
