import cvxpy as cp
import numpy as np 
import matplotlib.pyplot as plt

########################################################################################
# Systems of Linear Equations #
########################################################################################

A = np.array([[2,-4,2,-14],[-1,2,-2,11],[-1,2,-1,7]])
b = np.array([10.0,-6.0,-5.0])

def ConvProb_simple(A,b):
    x = A.T@(np.linalg.pinv(A@A.T))@b
    return x

x_star = ConvProb_simple(A,b)
print(x_star)


########################################################################################
########################################################################################

def proj(A,b,z):
    lambda_star = (np.linalg.pinv(A @ A.T))@(b - A @ z)
    return z + A.T @ lambda_star

def proj_grad_dec(A, b, x0, alpha, max_iter):
    x = np.array(x0).copy()
    x_star = ConvProb_simple(A,b)
    err = []
    iter = 0
    for t in range(max_iter):
        gradient = x
        x_new = x - alpha*gradient
        x_new = proj(A, b, x_new)
        error = np.linalg.norm(x_new - x_star, 2)
        err.append(error)
        iter += 1
        if error < 1e-10:
            break
        x = x_new
    return x, err, iter

step_sizes = [0.5,0.25,0.1,0.075,0.05,0.025]
x0 = np.array([1.0,1.0,1.0,1.0])
max_iter = 1000

plt.figure(figsize=(10, 8))
for step_size in step_sizes:
    x_final, errors, itr = proj_grad_dec(A, b, x0, step_size, max_iter)
    final_error = np.linalg.norm(x_final - x_star)
    plt.plot(errors, 'o-', label=f'Step size = {step_size}, Iterations = {itr}', linewidth=1.5, markersize=3)
    
    print(f"\nStep size: {step_size}")
    print(f"Final solution: {np.round(x_final, 5)}")
    print(f"Final Error ||x_f - x*||: {final_error:.5e}")
    print(f"Number of Iterations: {itr}")
    
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('$\|x(t) - x^*\|_2$', fontsize=12)
  
plt.title('Convergence of Projected Gradient Descent', fontsize=14)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('Q1.png')
plt.show()


########################################################################################
# SUPPORT VECTOR MACHINE #
########################################################################################

X = np.loadtxt("Data.csv", delimiter=",")
y = np.loadtxt("Labels.csv", delimiter=",")


########################################################################################
########################################################################################

n_samples, n_features = X.shape
w = cp.Variable(n_features)
b = cp.Variable()
obj = cp.Minimize((1/2) * cp.norm(w, 2)**2)
cons = [cp.multiply(y, X @ w + b) >= 1]
prob = cp.Problem(obj, cons)
prob.solve()

print("------------------------------------")
print("Primal Problem")
print("------------------------------------")
print("Optimal weights:", w.value)
print("Optimal bias:", b.value)
print("Optimal Primal:", prob.value)


########################################################################################
########################################################################################

def projection(z):
    if np.dot(y, y) == 0:
        return z
    proj = z - (np.dot(z, y) / np.dot(y, y))*y
    return np.maximum(0, proj)

def projected_gradient_descent(A, b, x0, alpha, max_itr):
    x = np.array(x0).copy()
    itr = max_itr
    for t in range(max_itr):
        x_new = projection(x - alpha*(A @ x + b))
        error = np.linalg.norm(x_new - x)
        if error < 1e-16:
            itr = t
            break
        x = x_new
    return x, itr

A = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    for j in range(n_samples):
        A[i,j] = y[i]*y[j]*np.dot(X[i], X[j])
A = (A + A.T) / 2
b = -np.ones(n_samples)
x0 = np.ones(n_samples)
xf, itr = projected_gradient_descent(A, b, x0, 1e-4, 100000)

print("\n------------------------------------")
print("Dual Problem")
print("------------------------------------")
print("Final values of lambda:", xf)

def dualprob(x):
    return 0.5 * np.dot(x, A @ x) + np.dot(x, b)
print("Dual Objective Function value:", -dualprob(xf))


########################################################################################
########################################################################################

w = np.sum(xf[:, None] * y[:, None] * X, axis=0)
print("\nWeight vector (w):", w)

tolerance = 1e-10
active_constraints = np.where(xf > tolerance)[0]
svm_index = active_constraints[0]
b = y[svm_index] - np.dot(w, X[svm_index])
print("Bias (b):", b)


########################################################################################
########################################################################################

gamma_p = np.sum(xf[y == 1])
gamma_n = np.sum(xf[y == -1])
print(f"\nSum of lambda values for y = +1: {gamma_p}")
print(f"Sum of lambda values for y = -1: {gamma_n}")
print(f"Gamma : {(gamma_p + gamma_n)/2}")
print("\n")


########################################################################################
########################################################################################
plt.figure(figsize=(10, 8))
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], marker='o', s=80, color='blue', label=r'$y=+1$')
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], marker='s', s=80, color='green', label=r'$y=-1$')
plt.scatter(X[active_constraints][:, 0], X[active_constraints][:, 1], 
            edgecolor='red', facecolor='none', s=150, linewidth=2, label=r'Active constraints')

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min = (-w[0] * x_min - b) / w[1]
y_max = (-w[0] * x_max - b) / w[1]
plt.plot([x_min, x_max], [y_min, y_max], 'k-', linewidth=2, label=r'Decision boundary ($w^\top x + b = 0$)')

plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend()
plt.grid()
plt.savefig("Q2.png")
plt.show()


########################################################################################
########################################################################################

plt.figure(figsize=(10, 8))
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], marker='o', s=80, color='blue', label=r'$y=+1$')
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], marker='s', s=80, color='green', label=r'$y=-1$')
plt.scatter(X[active_constraints][:, 0], X[active_constraints][:, 1], 
            edgecolor='red', facecolor='none', s=150, linewidth=2, label=r'Active constraints')

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.coolwarm)

y_min_line = (-w[0] * x_min - b) / w[1]
y_max_line = (-w[0] * x_max - b) / w[1]
plt.plot([x_min, x_max], [y_min_line, y_max_line], 'k-', linewidth=2, label=r'Decision boundary ($w^\top x + b = 0$)')

y_min_margin_1 = (-w[0] * x_min - b + 1) / w[1]
y_max_margin_1 = (-w[0] * x_max - b + 1) / w[1]
y_min_margin_2 = (-w[0] * x_min - b - 1) / w[1]
y_max_margin_2 = (-w[0] * x_max - b - 1) / w[1]

plt.plot([x_min, x_max], [y_min_margin_1, y_max_margin_1], 'k:', linewidth=1, label=r'Boundary ($w^\top x + b = +1$)')
plt.plot([x_min, x_max], [y_min_margin_2, y_max_margin_2], 'k--', linewidth=1, label=r'Boundary ($w^\top x + b = -1$)')

plt.fill_between(x=np.linspace(x_min, x_max, 100),
                 y1=(-w[0] * np.linspace(x_min, x_max, 100) - b + 1) / w[1],
                 y2=(-w[0] * np.linspace(x_min, x_max, 100) - b - 1) / w[1],
                 color='gray', alpha=0.5, label=r'Hyperplane margin')

plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend()
plt.grid(True)
plt.savefig("Q2_full.png")
plt.show()
