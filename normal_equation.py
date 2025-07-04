import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Theta = (X^T * X)^-1 * X^T * y
# np.transpose(X) = X.T
# np.matmul(X, y) = X @ y

def normal_equation(x, y):
    m = len(y)
    X = np.c_[np.ones((m, 1)), x]
    # X^T * X   
    X_T_X = X.T @ X 

    try:
        # (X^T * X)^-1
        X_T_X_inv = np.linalg.inv(X_T_X)
    except np.linalg.LinAlgError:  # fall back to a pseudo-inverse
        print("Warning: (X_T_X) is singular or near-singular.")
        return None 

    # X^T * y
    X_T_y = X.T @ y

    # (X^T * X)^-1 * X^T * y
    theta = X_T_X_inv @ X_T_y
    return theta

print("Select sample to test: ")
print("1) Single feature Prediction")
print("2) Car Price Prediction")
print("3) Housing Price Prediction")
print("Enter a number.")
choice = int(input())

df = None
x = None

match choice:
    case 1:
        np.random.seed(42)
        m = 100 # 100 data points
        x = 2 * np.random.rand(100, 1)
        y = 2 * x + 3 + np.random.randn(100, 1)
        
    case 2:
        df = pd.read_csv("samples/used_car_price_dataset.csv", sep=",")
        df = df.drop(columns = ['fuel_type', 'brand', 'transmission', 'color', 'service_history', "insurance_valid"], axis = 1)
        print(df.head(5))
        m = len(df)
        
        x = df.drop(columns = ["price_usd"], axis = 1)
        y = df["price_usd"]
    case 3:
        df = pd.read_csv("samples/housing_price_dataset.csv", sep = ",")
        df = df.drop(columns = ['Neighborhood'], axis = 1)
        print(df.head(5))
        m = len(df)

        x = df.drop(columns = ["price_usd"], axis = 1)
        y = df["price_usd"]
    case _:
        print("Unknown selection")
        quit()

theta = normal_equation(x, y)
print(theta)

# Small sample choice w/ graph
if choice == 1:
    plt.plot(x, y, linestyle = "none", marker = ".")
    plt.title(f"y = {theta[0][0]} + {theta[1][0]}x")
    plt.ylabel('X-axis')
    plt.ylabel('Y-axis')
    regression = theta[0][0] + x * theta[1][0]
    plt.plot(x, regression)
    plt.show()
else:
    for i in range(len(x.columns)):
        print(f"Theta for {x.columns[i]} is {theta[i + 1]}")

    print(f"The equation is: y = {theta[0]}", end = "")
    for i in range(len(x.columns)):
        print(f" + {theta[i + 1]} * x_{i + 1}", end = "")

    