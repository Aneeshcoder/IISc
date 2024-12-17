from CMO_A1 import f1, f2, f3, f4
import numpy as np
import matplotlib.pyplot as plt

#Name: Aneesh Panchal
sr_num = 25223

##############################################################################################################
'''Question 1 Part 1'''
print("\n\nQuestion 1 Part 1\n\n")
##############################################################################################################

def isConvex(func, int_start, int_end):
    x = np.linspace(int_start, int_end, 10000)
    fx = np.array([func(sr_num, xval) for xval in x])
    plt.plot(x, fx)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.show()
    diff_val = np.diff(fx)
    flag = -1
    for i in range(len(diff_val)):
        if diff_val[i] > 0:
            flag = 1
        elif diff_val[i] < 0:
            flag = -1
        if (flag == 1 and diff_val[i] < 0) or (flag == -1 and diff_val[i] > 0):
            return False
    return True

def isStrictConvex(func, int_start, int_end):
    x = np.linspace(int_start, int_end, 10000)
    fx = np.array([func(sr_num, xval) for xval in x])
    diff_val = np.diff(fx)
    flag = -1
    for i in range(len(diff_val)):
        if diff_val[i] > 0:
            flag = 1
        elif diff_val[i] < 0:
            flag = -1
        # Extra condition for strict convexity 
        elif diff_val[i] == 0:  
            return False
        if (flag == 1 and diff_val[i] < 0) or (flag == -1 and diff_val[i] > 0):
            return False
    return True

def findMin(func, int_start, int_end):
    x = np.linspace(int_start, int_end, 10)
    fx = np.array([func(sr_num, xval) for xval in x])
    diff_val = np.diff(fx)
    xval = []
    flag = True
    for i in range(len(diff_val)):
        if flag == True and diff_val[i] > 0:
            xval.append(x[i])
            flag = False
        if diff_val[i] > 0:
            continue
        elif diff_val[i] < 0:
            continue
        else:
            xval.append(x[i])
    return xval

Convexf1  = isConvex(f1, -2, 2)
print("Function f1 is convex:", Convexf1)
Convexf2 = isConvex(f2, -2, 2)
print("Function f2 is convex:", Convexf2)

StrictConvexf1 = isStrictConvex(f1, -2, 2)
print("Function f1 is strictly convex:", StrictConvexf1)
StrictConvexf2 = isStrictConvex(f2, -2, 2)
print("Function f2 is strictly convex:", StrictConvexf2)

Minf1 = findMin(f1, -2, 2)
if len(Minf1) == 1:
    print("Minimum value of x is:", Minf1[0])
    print("Minimum value of f1(x) is:", f1(sr_num, Minf1[0]))
else:
    print("Multiple minima found with min value of x: ", Minf1[0])
    print("Multiple minima found with max value of x: ", Minf1[-1])
    print("Minimum values of f1(x) is: ", f1(sr_num, Minf1[0]))

Minf2 = findMin(f2, -2, 2) 
if len(Minf2) == 1:
    print("Minimum value of x is:", Minf2[0])
    print("Minimum value of f2(x) is:", f2(sr_num, Minf2[0]))
else:
    print("Multiple minima found with min value of x: ", Minf2[0])
    print("Multiple minima found with max value of x: ", Minf2[-1])
    print("Minimum values of f2(x) is: ", f2(sr_num, Minf2[0]))


##############################################################################################################
'''Question 1 Part 2'''
print("\n\nQuestion 1 Part 2\n\n")
##############################################################################################################

def isCoercive(func):
    x = np.linspace(1, 5, 5)
    fx = np.array([func(sr_num, xval) for xval in x])
    degree = 4
    coefficients = np.polyfit(x, fx, degree)
    if coefficients[0] > 0:
        return True
    else:
        return False

# Assumption: Condition of quartic behaviour is known
def FindStationaryPoints(func):
    x = np.linspace(1, 5, 5)
    fx = np.array([func(sr_num, xval) for xval in x])
    degree = 4
    coefficients = np.polyfit(x, fx, degree)
    print(coefficients)
    polynomial = np.poly1d(coefficients)
    FirstDiff = polynomial.deriv()
    SecondDiff = FirstDiff.deriv()
    StationaryPoints = np.roots(FirstDiff)
    Roots = np.roots(coefficients)
    results = {'Roots': [], 'Minima': [], 'LocalMaxima': []}
    for point in Roots:
        results['Roots'].append(point)
    for point in StationaryPoints:
        if np.isreal(point) and SecondDiff(point) > 0:
            results['Minima'].append(point)
        elif np.isreal(point) and SecondDiff(point) < 0:
            results['LocalMaxima'].append(point)
    for key in results:
        results[key] = sorted(results[key])
    return results

Coercive = isCoercive(f3)
print("Function is coercive:", Coercive)

StationaryPoints = FindStationaryPoints(f3)
print("Roots:", StationaryPoints['Roots'])
print("Minima:", StationaryPoints['Minima'])
print("Local Maxima:", StationaryPoints['LocalMaxima'])


##############################################################################################################
'''Plotting function required for Question 2'''
##############################################################################################################

def plots(results):
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    axs[0, 0].plot(results['grad_norms'])
    axs[0, 0].set_title(r'Gradient Norm $||grad(f(x_k))||_2$')
    axs[0, 0].set_xlabel('Iteration')
    axs[0, 0].set_ylabel(r'$||grad(f(x_k))||_2$')
    axs[0, 0].grid(True)

    axs[0, 1].plot(results['func_diffs'])
    axs[0, 1].set_title(r'Function Value Difference $f(x_k) - f(x_T)$')
    axs[0, 1].set_xlabel('Iteration')
    axs[0, 1].set_ylabel(r'$f(x_k) - f(x_T)$')
    axs[0, 1].grid(True)

    axs[1, 0].plot(results['func_diff_ratios'])
    axs[1, 0].set_title(r'Ratio of Function Value Differences $\frac{f(x_k) - f(x_T)}{f(x_{k-1}) - f(x_T)}$')
    axs[1, 0].set_xlabel('Iteration')
    axs[1, 0].set_ylabel(r'$\frac{f(x_k) - f(x_T)}{f(x_{k-1}) - f(x_T)}$')
    axs[1, 0].grid(True)

    axs[1, 1].plot(results['distance_squares'])
    axs[1, 1].set_title(r'Squared Distance $||x_k - x_T||^2_2$')
    axs[1, 1].set_xlabel('Iteration')
    axs[1, 1].set_ylabel(r'$||x_k - x_T||^2_2$')
    axs[1, 1].grid(True)

    axs[2, 0].plot(results['distance_ratios'])
    axs[2, 0].set_title(r'Ratio of Squared Distance $\frac{||x_k - x_T||^2_2}{||x_{k-1} - x_T||^2_2}$')
    axs[2, 0].set_xlabel('Iteration')
    axs[2, 0].set_ylabel(r'$\frac{||x_k - x_T||^2_2}{||x_{k-1} - x_T||^2_2}$')
    axs[2, 0].grid(True)
    
    axs[2, 1].axis('off')
    plt.tight_layout()
    plt.show()


##############################################################################################################
'''Question 2 Part (a)'''
print("\n\nQuestion 2 Part (a)\n\n")
##############################################################################################################

def ConstantGradientDescent(alpha, initialx):
    x = np.array([initialx])
    fx, gradfx = f4(sr_num, x[0])
    fx_values = np.array([fx])
    grad_norms = np.array([np.linalg.norm(gradfx)])
    func_diffs = []
    distance_squares = []
    func_diff_ratios = []
    distance_ratios = []
    for i in range(10000):
        fx, gradfx = f4(sr_num, x[i])
        x_new = x[i] - alpha*gradfx
        x = np.append(x,[x_new],axis=0)
        
        fx, gradfx = f4(sr_num, x[i+1])
        grad_norm = np.linalg.norm(gradfx)
        fx_values = np.append(fx_values, [fx])
        grad_norms = np.append(grad_norms, [grad_norm])
    x_T = x[x.shape[0]-1]
    print("x*: ", x[x.shape[0]-1])
    print("f(x*): ", fx_values[fx_values.__len__()-1])
    print("Iterations: ", x.shape[0]-1)
    
    for i in range(len(x)):
        fx, gradval = f4(sr_num, x[i])
        grad_norm = np.linalg.norm(gradval)
        func_diff = fx - f4(sr_num, x_T)[0]
        distance_square = np.linalg.norm(x[i] - x_T) ** 2
        func_diffs.append(func_diff)
        distance_squares.append(distance_square)
        if i > 0:
            func_diff_ratios.append(func_diffs[i] / func_diffs[i-1] if func_diffs[i-1] != 0 else 0)
            distance_ratios.append(distance_squares[i] / distance_squares[i-1] if distance_squares[i-1] != 0 else 0)

    return {
        'grad_norms': grad_norms,
        'func_diffs': func_diffs,
        'func_diff_ratios': func_diff_ratios,
        'distance_squares': distance_squares,
        'distance_ratios': distance_ratios
    }
    
print("\n\nConstantGradientDescent\n")
plots(ConstantGradientDescent(1e-5, [0.0,0.0,0.0,0.0,0.0]))


##############################################################################################################
'''Question 2 Part (b)'''
print("\n\nQuestion 2 Part (b)\n\n")
##############################################################################################################

def DiminishingGradientDescent(InitialAlpha, initialx):
    x = np.array([initialx])
    fx, gradfx = f4(sr_num, x[0])
    fx_values = np.array([fx])
    grad_norms = np.array([np.linalg.norm(gradfx)])
    func_diffs = []
    distance_squares = []
    func_diff_ratios = []
    distance_ratios = []
    alpha = InitialAlpha
    for i in range(10000):
        fx, gradfx = f4(sr_num, x[i])
        alpha = InitialAlpha/(i+1)
        x_new = x[i] - alpha*gradfx
        x = np.append(x,[x_new],axis=0)
        
        fx, gradfx = f4(sr_num, x[i+1])
        grad_norm = np.linalg.norm(gradfx)
        fx_values = np.append(fx_values, [fx])
        grad_norms = np.append(grad_norms, [grad_norm])
    x_T = x[x.shape[0]-1]
    print("x*: ", x[x.shape[0]-1])
    print("f(x*): ", fx_values[fx_values.__len__()-1])
    print("Iterations: ", x.shape[0]-1)
    
    for i in range(len(x)):
        fx, gradval = f4(sr_num, x[i])
        grad_norm = np.linalg.norm(gradval)
        func_diff = fx - f4(sr_num, x_T)[0]
        distance_square = np.linalg.norm(x[i] - x_T) ** 2
        func_diffs.append(func_diff)
        distance_squares.append(distance_square)
        if i > 0:
            func_diff_ratios.append(func_diffs[i] / func_diffs[i-1] if func_diffs[i-1] != 0 else 0)
            distance_ratios.append(distance_squares[i] / distance_squares[i-1] if distance_squares[i-1] != 0 else 0)

    return {
        'grad_norms': grad_norms,
        'func_diffs': func_diffs,
        'func_diff_ratios': func_diff_ratios,
        'distance_squares': distance_squares,
        'distance_ratios': distance_ratios
    }
        
print("\n\nDiminishingGradientDescent\n")
plots(DiminishingGradientDescent(1e-3, [0.0,0.0,0.0,0.0,0.0]))


##############################################################################################################
'''Question 2 Part (c)'''
print("\n\nQuestion 2 Part (c)\n\n")
##############################################################################################################

def FindAlphaInExact(x, InitialAlpha, c1, c2, gamma):
    alpha = InitialAlpha
    fx, gradfx = f4(sr_num, x)
    new_x = x - alpha*gradfx
    fx_new, gradfx_new = f4(sr_num, new_x)
    iter = 0
    while (fx_new > fx - c1*alpha*(gradfx.T)@gradfx) or ((gradfx.T)@gradfx_new > c2*(gradfx.T)@gradfx) and alpha !=0:
        alpha = alpha*gamma
        new_x = x - alpha*gradfx
        fx_new, gradfx_new = f4(sr_num, new_x)
        iter += 1
    return alpha
    
def InExactLineSearch(c1, c2, gamma):
    x = np.array([[0.0,0.0,0.0,0.0,0.0]])
    fx, gradfx = f4(sr_num, x[0])
    fx_values = np.array([fx])
    grad_norms = np.array([np.linalg.norm(gradfx)])
    func_diffs = []
    distance_squares = []
    func_diff_ratios = []
    distance_ratios = []
    for i in range(10000):
        fx, gradfx = f4(sr_num, x[i])
        alpha = FindAlphaInExact(x[i], 1, c1, c2, gamma)
        x_new = x[i] - alpha*gradfx
        x = np.append(x,[x_new],axis=0)
        fx = f4(sr_num, x[i+1])[0]
        grad_norm = np.linalg.norm(gradfx)
        fx_values = np.append(fx_values, [fx])
        grad_norms = np.append(grad_norms, [grad_norm])
        
        #Termination Condition
        if alpha == 0:
            break
    
    x_T = x[x.shape[0]-1]
    print("x*: ", x[x.shape[0]-1])
    print("f(x*): ", fx_values[fx_values.__len__()-1])
    print("Iterations: ", x.shape[0]-1)
    
    for i in range(len(x)):
        fx, gradval = f4(sr_num, x[i])
        grad_norm = np.linalg.norm(gradval)
        
        func_diff = fx - f4(sr_num, x_T)[0]
        distance_square = np.linalg.norm(x[i] - x_T) ** 2
        func_diffs.append(func_diff)
        distance_squares.append(distance_square)
        if i > 0:
            func_diff_ratios.append(func_diffs[i] / func_diffs[i-1] if func_diffs[i-1] != 0 else 0)
            distance_ratios.append(distance_squares[i] / distance_squares[i-1] if distance_squares[i-1] != 0 else 0)

    return {
        'grad_norms': grad_norms,
        'func_diffs': func_diffs,
        'func_diff_ratios': func_diff_ratios,
        'distance_squares': distance_squares,
        'distance_ratios': distance_ratios
    }
    
print("\n\nInExactLineSearch\n")
plots(InExactLineSearch(0.5,0.5,0.5))


##############################################################################################################
'''Question 2 Part (d)'''
print("\n\nQuestion 2 Part (d)\n\n")
##############################################################################################################

def FindAlphaExact(x):
    x0 = np.array([0.0,0.0,0.0,0.0,0.0])
    fx0, gradfx0 = f4(sr_num, x0)
    fx, px = f4(sr_num, x)
    fp, gradfpx = f4(sr_num, -px)
    num = (px.T)@px
    deno = 2*(fp + gradfx0.T@px)
    alpha = num/deno
    return alpha

def ExactLineSearch():
    x = np.array([[0.0,0.0,0.0,0.0,0.0]])
    fx, gradfx = f4(sr_num, x[0])
    fx_values = np.array([fx])
    grad_norms = np.array([np.linalg.norm(gradfx)])
    func_diffs = []
    distance_squares = []
    func_diff_ratios = []
    distance_ratios = []
    for i in range(10000):
        fx, gradfx = f4(sr_num, x[i])
        alpha = FindAlphaExact(x[i])
        x_new = x[i] - alpha*gradfx
        x = np.append(x,[x_new],axis=0)
        fx = f4(sr_num, x[i+1])[0]
        grad_norm = np.linalg.norm(gradfx)
        fx_values = np.append(fx_values, [fx])
        grad_norms = np.append(grad_norms, [grad_norm])
    
    x_T = x[x.shape[0]-1]
    print("x*: ", x[x.shape[0]-1])
    print("f(x*): ", fx_values[fx_values.__len__()-1])
    print("Iterations: ", x.shape[0]-1)
    
    for i in range(len(x)):
        fx, gradval = f4(sr_num, x[i])
        grad_norm = np.linalg.norm(gradval)
        
        func_diff = fx - f4(sr_num, x_T)[0]
        distance_square = np.linalg.norm(x[i] - x_T) ** 2
        func_diffs.append(func_diff)
        distance_squares.append(distance_square)
        if i > 0:
            func_diff_ratios.append(func_diffs[i] / func_diffs[i-1] if func_diffs[i-1] != 0 else 0)
            distance_ratios.append(distance_squares[i] / distance_squares[i-1] if distance_squares[i-1] != 0 else 0)

    return {
        'grad_norms': grad_norms,
        'func_diffs': func_diffs,
        'func_diff_ratios': func_diff_ratios,
        'distance_squares': distance_squares,
        'distance_ratios': distance_ratios
    }
    
print("\n\nExactLineSearch\n")
plots(ExactLineSearch())


##############################################################################################################
'''Function Definitions and Plotting Function for Question 3 Part 3, 4, 5'''
##############################################################################################################

def func(x, y):
    return np.exp(x*y)

def fgrad(x, y):
    df_dx = y * np.exp(x * y)
    df_dy = x * np.exp(x * y)
    return np.array([df_dx, df_dy])

def plottingFunc(trajectory, function_values, iter):
    x = np.linspace(min(trajectory[:, 0]) - 1, max(trajectory[:, 0]) + 1, 100)
    y = np.linspace(min(trajectory[:, 1]) - 1, max(trajectory[:, 1]) + 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    contour = plt.contour(X, Y, Z, levels=100, cmap='viridis')
    plt.colorbar(contour)
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', markersize=5, label='Trajectory')
    plt.title('Contour Plot with Gradient Descent Trajectory')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(iter + 1), function_values, 'bo-')
    plt.title('Function Value over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel(r'$f(x)$')

    plt.tight_layout()
    plt.show()


##############################################################################################################
'''Question 3 Part 3'''
print("\n\nQuestion 3 Part 3\n\n")
##############################################################################################################

x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)
Z = func(X, Y)

plt.figure(figsize=(8, 6))
contour = plt.contour(X, Y, Z, levels=100, cmap='viridis')
plt.colorbar(contour)
plt.title(r"Contour Plot of $f(x, y) = e^{xy}$")
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
plt.show()


##############################################################################################################
'''Question 3 Part 4'''
print("\n\nQuestion 3 Part 4\n\n")
##############################################################################################################

def FixedStep(init_x, init_y, alpha, iter):
    x, y = init_x, init_y
    trajectory = [(x, y)]
    func_val = [func(x, y)]
    
    for i in range(iter):
        grad = fgrad(x, y)
        x -= alpha * grad[0]
        y -= alpha * grad[1]
        trajectory.append((x, y))
        func_val.append(func(x, y))
    
    return np.array(trajectory), func_val

init_x, init_y = 1, 1
alpha = 1e-1
iter = 10000
trajectory, function_values = FixedStep(init_x, init_y, alpha, iter)
plottingFunc(trajectory, function_values, iter)


##############################################################################################################
'''Question 3 Part 5'''
print("\n\nQuestion 3 Part 5\n\n")
##############################################################################################################

def DecreasingStep(init_x, init_y, alpha, iter):
    x, y = init_x, init_y
    trajectory = [(x, y)]
    func_val = [func(x, y)]
    alpha_0 = alpha
    for i in range(iter):
        grad = fgrad(x, y)
        alpha = alpha_0 / (i + 1)
        x -= alpha * grad[0]
        y -= alpha * grad[1]
        trajectory.append((x, y))
        func_val.append(func(x, y))
    
    return np.array(trajectory), func_val

init_x, init_y = 1, 1
alpha = 1e-1
iter = 10000
trajectory, function_values = DecreasingStep(init_x, init_y, alpha, iter)
plottingFunc(trajectory, function_values, iter)


##############################################################################################################
'''Extra Functions and Plotting Function for Question 3 Part 6, 7, 8, 9'''
##############################################################################################################

def NormalDistrib():
    mean = [0, 0]
    var1 = 1
    var2 = 1
    cov = [[var1, 0], [0, var2]]
    samples = np.random.multivariate_normal(mean, cov, size = 1)
    return samples[0][0], samples[0][1]

def NormalDistribVariable(iter):
    mean = [0, 0]
    var1 = 1/(iter+1)
    var2 = 1/(iter+1)
    cov = [[var1, 0], [0, var2]]
    samples = np.random.multivariate_normal(mean, cov, size = 1)
    return samples[0][0], samples[0][1]

def PlottingVariance(means, std_devs, trajectories):
    iterations = np.arange(1, len(means) + 1) - 1
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    axs[0].errorbar(x=iterations, y=means, yerr=std_devs, fmt='o', color='blue', 
                    label='Expected Function Value with Std Dev', capsize=5, alpha=0.7)
    
    axs[0].plot(iterations, means, color='red', label='Expected Function Value')
    axs[0].fill_between(iterations, means - std_devs, means + std_devs, color='blue', alpha=0.2)
    axs[0].set_title('Expected Function Value with Error Bars')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel(r'$f(x)$')
    axs[0].grid(True)
    axs[0].legend()

    for traj in trajectories:
        axs[1].plot(traj[:, 0], traj[:, 1], color='black', linestyle='-')
    x_min = min([traj[:, 0].min() for traj in trajectories])
    x_max = max([traj[:, 0].max() for traj in trajectories])
    y_min = min([traj[:, 1].min() for traj in trajectories])
    y_max = max([traj[:, 1].max() for traj in trajectories])
    
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)
    contour = axs[1].contourf(X, Y, Z, cmap='viridis')
    fig.colorbar(contour, ax=axs[1])
    
    axs[1].set_xlim(x_min, x_max)
    axs[1].set_ylim(y_min, y_max)
    axs[1].set_title('Trajectories of All Tests')
    axs[1].set_xlabel(r'$x$')
    axs[1].set_ylabel(r'$y$')
    axs[1].grid(True)
    plt.tight_layout()
    plt.show()


##############################################################################################################
'''Question 3 Part 6'''
print("\n\nQuestion 3 Part 6\n\n")
##############################################################################################################

def FixStepFixNoise(init_x, init_y, alpha, iter):
    x, y = init_x, init_y
    trajectory = [(x, y)]
    func_val = [func(x, y)]
    
    for i in range(iter):
        grad = fgrad(x, y)
        chix, chiy = NormalDistrib()
        x -= alpha * (grad[0] + chix)
        y -= alpha * (grad[1] + chiy)
        trajectory.append((x, y))
        func_val.append(func(x, y))
    
    return np.array(trajectory), func_val

init_x, init_y = 1, 1
alpha = 1e-1
iter = 100
num_experiments = 100
means = []
std_devs = []
trajectories = []
values = []
for i in range(num_experiments):
    trajectory, function_values = FixStepFixNoise(init_x, init_y, alpha, iter)
    values.append(function_values)
    trajectories.append(trajectory)

means = np.mean(values, axis=0)
std_devs = np.std(values, axis=0)
means = np.array(means)
std_devs = np.array(std_devs)
PlottingVariance(means, std_devs, trajectories)


##############################################################################################################
'''Question 3 Part 7'''
print("\n\nQuestion 3 Part 7\n\n")
##############################################################################################################

def FixStepDecNoise(init_x, init_y, alpha, iter):
    x, y = init_x, init_y
    trajectory = [(x, y)]
    func_val = [func(x, y)]
    
    for i in range(iter):
        grad = fgrad(x, y)
        chix, chiy = NormalDistribVariable(i)
        x -= alpha * (grad[0] + chix)
        y -= alpha * (grad[1] + chiy)
        trajectory.append((x, y))
        func_val.append(func(x, y))
    
    return np.array(trajectory), func_val

init_x, init_y = 1, 1
alpha = 1e-1
iter = 100
num_experiments = 100
means = []
std_devs = []
trajectories = []
values = []
for i in range(num_experiments):
    trajectory, function_values = FixStepDecNoise(init_x, init_y, alpha, iter)
    values.append(function_values)
    trajectories.append(trajectory)

means = np.mean(values, axis=0)
std_devs = np.std(values, axis=0)
means = np.array(means)
std_devs = np.array(std_devs)
PlottingVariance(means, std_devs, trajectories)


##############################################################################################################
'''Question 3 Part 8'''
print("\n\nQuestion 3 Part 8\n\n")
##############################################################################################################

def DecStepFixNoise(init_x, init_y, alpha, iter):
    x, y = init_x, init_y
    trajectory = [(x, y)]
    func_val = [func(x, y)]
    alpha_0 = alpha
    for i in range(iter):
        grad = fgrad(x, y)
        chix, chiy = NormalDistrib()
        alpha = alpha_0 / (i + 1)
        x -= alpha * (grad[0] + chix)
        y -= alpha * (grad[1] + chiy)
        trajectory.append((x, y))
        func_val.append(func(x, y))
    
    return np.array(trajectory), func_val

init_x, init_y = 1, 1
alpha = 1e-1
iter = 100
num_experiments = 100
means = []
std_devs = []
trajectories = []
values = []
for i in range(num_experiments):
    trajectory, function_values = DecStepFixNoise(init_x, init_y, alpha, iter)
    values.append(function_values)
    trajectories.append(trajectory)

means = np.mean(values, axis=0)
std_devs = np.std(values, axis=0)
means = np.array(means)
std_devs = np.array(std_devs)
PlottingVariance(means, std_devs, trajectories)


##############################################################################################################
'''Question 3 Part 9'''
print("\n\nQuestion 3 Part 9\n\n")
##############################################################################################################

def DecStepDecNoise(init_x, init_y, alpha, iter):
    x, y = init_x, init_y
    trajectory = [(x, y)]
    func_val = [func(x, y)]
    alpha_0 = alpha
    for i in range(iter):
        grad = fgrad(x, y)
        chix, chiy = NormalDistribVariable(i)
        alpha = alpha_0 / (i + 1)
        x -= alpha * (grad[0] + chix)
        y -= alpha * (grad[1] + chiy)
        trajectory.append((x, y))
        func_val.append(func(x, y))
    
    return np.array(trajectory), func_val

init_x, init_y = 1, 1
alpha = 1e-1
iter = 100
num_experiments = 100
means = []
std_devs = []
trajectories = []
values = []
for i in range(num_experiments):
    trajectory, function_values = DecStepDecNoise(init_x, init_y, alpha, iter)
    values.append(function_values)
    trajectories.append(trajectory)

means = np.mean(values, axis=0)
std_devs = np.std(values, axis=0)
means = np.array(means)
std_devs = np.array(std_devs)
PlottingVariance(means, std_devs, trajectories)


##############################################################################################################
'''Function definition and Plotting Function for Question 4 Part 2 (a), (b)'''
##############################################################################################################

def func(x):
    return x*(x-1)*(x-3)*(x+2)

def plotting(func_a_vals, func_b_vals, interval_vals, interval_ratios):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    axs[0].plot(func_a_vals, label=r'$f(a)$')
    axs[0].plot(func_b_vals, label=r'$f(b)$')
    axs[0].set_title('Function Values at a and b')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Function Value')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(interval_vals, label=r'$b - a$')
    axs[1].set_title('Interval Lengths')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Length')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(interval_ratios, label=r'$\frac{b_t - a_t}{b_{t-1} - a_{t-1}}$')
    axs[2].set_title(r'Ratio of Interval Lengths  $\frac{b_t - a_t}{b_{t-1} - a_{t-1}}$')
    axs[2].set_xlabel('Iteration')
    axs[2].set_ylabel(r'$\frac{b_t - a_t}{b_{t-1} - a_{t-1}}$')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


##############################################################################################################
'''Question 4 Part 2 (a)'''
print("\n\nQuestion 4 Part 2 (a)\n\n")
##############################################################################################################

def GoldenSectionSearch(a, b, threshold):
    phi = (1 + np.sqrt(5)) / 2
    rho = phi - 1
    x1 = rho*a + (1-rho)*b
    x2 = rho*b + (1-rho)*a
    iteration = 0
    a_vals = [a]
    b_vals = [b]
    func_a_vals = [func(a)]
    func_b_vals = [func(b)]
    interval_vals = [b - a]
    
    while abs(b - a) > threshold:
        if func(x1) <= func(x2):
            b = x2
            x2 = x1
            x1 = rho*a + (1-rho)*b
        else:
            a = x1
            x1 = x2
            x2 = rho*b + (1-rho)*a
        
        a_vals.append(a)
        b_vals.append(b)
        func_a_vals.append(func(a))
        func_b_vals.append(func(b))
        interval_vals.append(b - a)   
        iteration += 1
    
    interval_ratios = [interval_vals[i] / interval_vals[i-1] if i > 0 else 1 for i in range(len(interval_vals))]
    return (a + b) / 2, iteration, func_a_vals, func_b_vals, interval_vals, interval_ratios, a_vals, b_vals

a, b = 1, 3
threshold = 1e-4
xsol, iter, func_a_vals, func_b_vals, interval_vals, interval_ratios, a_vals, b_vals = GoldenSectionSearch(a, b, threshold)
print(f"Solution: {xsol}, Function value: {func(xsol)}, Iterations: {iter}")
for i in range(len(func_a_vals)):
    print(f"Iteration {i}: a = {a_vals[i]}, b = {b_vals[i]}, interval = {interval_vals[i]}")
plotting(func_a_vals, func_b_vals, interval_vals, interval_ratios)


##############################################################################################################
'''Question 4 Part 2 (b)'''
print("\n\nQuestion 4 Part 2 (b)\n\n")
##############################################################################################################

def fib(n):
    if n == 0:
        return 1
    elif n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)

def FibonacciSearch(a, b, threshold):
    n = 1
    while (b-a)/threshold > fib(n+1):
        n += 1
    
    iteration = 0
    rho = 1 - (fib(n - iteration)/fib(n - iteration + 2))
    x1 = rho*a + (1-rho)*b
    x2 = rho*b + (1-rho)*a
    a_vals = [a]
    b_vals = [b]
    func_a_vals = [func(a)]
    func_b_vals = [func(b)]
    interval_vals = [b - a]
    
    for i in range(1, n+1):
        rho = 1 - (fib(n - i)/fib(n - i + 2))
        if func(x1) <= func(x2):
            b = x2
            x2 = x1
            x1 = rho*a + (1-rho)*b
        else:
            a = x1
            x1 = x2
            x2 = rho*b + (1-rho)*a
        
        a_vals.append(a)
        b_vals.append(b)
        func_a_vals.append(func(a))
        func_b_vals.append(func(b))
        interval_vals.append(b - a)        
        iteration += 1
    
    interval_ratios = [interval_vals[i] / interval_vals[i-1] if i > 0 else 1 for i in range(len(interval_vals))]
    return (a + b) / 2, iteration, func_a_vals, func_b_vals, interval_vals, interval_ratios

a, b = 1, 3
threshold = 1e-4
xsol, iter, func_a_vals, func_b_vals, interval_vals, interval_ratios = FibonacciSearch(a, b, threshold)
print(f"Solution: {xsol}, Function value: {func(xsol)}, Iterations: {iter}")
for i in range(len(func_a_vals)):
    print(f"Iteration {i}: a = {a_vals[i]}, b = {b_vals[i]}, interval = {interval_vals[i]}")
plotting(func_a_vals, func_b_vals, interval_vals, interval_ratios)
