import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

X, Y = load_diabetes(return_X_y = True)
feature_names = load_diabetes()['feature_names']
X = np.hstack((np.ones((X.shape[0], 1)), X))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

def mse_error(X,Y,w):
    w = np.array(w)
    N = X.shape[0]
    return ((np.sum(Y - X @ w)**2)/N)
def OrdinaryLeastSquares(X, Y):
    (n, d) = X.shape
    w = cp.Variable(d)
    objective = cp.Minimize(cp.sum((Y - X @ w)**2))
    problem = cp.Problem(objective)
    problem.solve()
    return np.array(w.value),mse_error(X,Y,w.value)
def LassoRegression(X, Y, lamda):
    (n, d) = X.shape
    w = cp.Variable(d)
    objective = cp.Minimize(cp.sum((Y - X @ w)**2) + lamda*cp.norm(w, 1))
    problem = cp.Problem(objective)
    problem.solve()
    return np.array(w.value),mse_error(X,Y,w.value)
def RidgeRegression(X, Y, lamda):
    (n, d) = X.shape
    w = cp.Variable(d)
    objective = cp.Minimize(cp.sum((Y - X @ w)**2) + lamda*cp.norm(w, 2))
    problem = cp.Problem(objective)
    problem.solve()
    return np.array(w.value),mse_error(X, Y, w.value)

w_opt, train_error = OrdinaryLeastSquares(X_train, Y_train)
test_error = mse_error(X_test, Y_test, w_opt)
print("weight vector = ", w_opt)
print("train error = ", train_error)
print("test error = ", test_error)

w_opt, train_error = LassoRegression(X_train, Y_train, 100)
test_error = mse_error(X_test, Y_test, w_opt)
print("weight vector = ", w_opt)
print("train error = ", train_error)
print("test error = ", test_error)

w_opt, train_error = RidgeRegression(X_train, Y_train, 100)
test_error = mse_error(X_test, Y_test, w_opt)
print("weight vector = ", w_opt)
print("train error = ", train_error)
print("test error = ", test_error)

lambdas = np.array([0.1, 0.3, 1, 3, 5, 10, 30, 100, 300,1000, 3000])

weights = []
train_errors = []
test_errors_arr = []
for l in lambdas:
    w,e = LassoRegression(X,Y,l)
    test_error = mse_error(X_test,Y_test,w)
    test_errors_arr.append(test_error)
    train_errors.append(e)
    weights.append(w)
weights = np.array(weights)
train_errors= np.array(train_errors)
test_errors_arr = np.array(test_errors_arr)
for i in range(weights.shape[1]):
    plt.plot(lambdas,weights[:,i],'-*',label=f"w_{i}")
plt.xlabel("lambdas")
plt.ylabel("weights")
plt.xscale("log")
plt.legend(bbox_to_anchor=(1.04,1),loc = "upper left")
plt.show()

plt.plot(lambdas,train_errors , label = "train errors")
plt.plot(lambdas,test_errors_arr,label = "test error")
plt.xlabel("lambas")
plt.ylabel("errors")
plt.xscale("log")
plt.legend()
plt.show()


weights = []
train_errors = []
test_errors_arr = []
for l in lambdas:
    w,e = RidgeRegression(X_train,Y_train,l)
    test_error = mse_error(X_test,Y_test,w)
    test_errors_arr.append(test_error)
    train_errors.append(e)
    weights.append(w)
weights = np.array(weights)
train_errors= np.array(train_errors)
test_errors_arr = np.array(test_errors_arr)
for i in range(weights.shape[1]):
    plt.plot(lambdas,weights[:,i],'-*',label=f"w_{i}")
plt.xlabel("lambdas")
plt.ylabel("weights")
plt.xscale("log")
plt.legend(bbox_to_anchor=(1.04,1),loc = "upper left")
plt.show()

plt.plot(lambdas,train_errors , label = "train errors")
plt.plot(lambdas,test_errors_arr,label = "test error")
plt.xlabel("lambas")
plt.ylabel("errors")
plt.legend()
plt.show()

