import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Preparing the data
# Features (Time Spent, Products Viewed)
X = np.array([
    [20, 5],
    [35, 8],
    [25, 10],
    [40, 12],
    [55, 15],
    [60, 18],
    [70, 20],
    [80, 22],
    [45, 14],
    [30, 7]
])

# Labels (0 for No Subscription, 1 for Yes Subscription)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0])

# training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Create and Train the SVM Model
# Linear SVM (SVC stands for Support Vector Classifier)
model = svm.SVC(kernel='linear', C=1.0)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 4. Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")

# 5. Visualize the Decision Boundary (for 2D data)

plt.figure(figsize=(10, 7))

# Plot the training data points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=100, cmap='coolwarm', edgecolors='k', label='Training Data')

plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=150, marker='X', cmap='coolwarm', edgecolors='k', label='Test Data')


# Plot the decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)

# Plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# Plot support vectors
ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=200, linewidth=1, facecolors='none', edgecolors='blue', label='Support Vectors')

plt.xlabel("Time Spent on Website (minutes)")
plt.ylabel("Number of Products Viewed")
plt.title("SVM Decision Boundary for Customer Subscription Prediction")
plt.legend()
plt.grid(True)
plt.show()

# 6. Predict for New Customers
print("\n--- Predicting for New Customers ---")
new_customers = np.array([
    [75, 25],
    [22, 6],
    [50, 16]
])

predictions = model.predict(new_customers)

for i, cust in enumerate(new_customers):
    pred_label = "Yes (Premium Subscription)" if predictions[i] == 1 else "No (No Subscription)"
    print(f"Customer with Time Spent={cust[0]} and Products Viewed={cust[1]} -> Prediction: {pred_label}")