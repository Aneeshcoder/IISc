import numpy as np
import matplotlib.pyplot as plt
from oracles_updated import f1, f2, f3

#Name: Aneesh Panchal
sr_num = 25223

##############################################################################################################
'''Question 1'''
##############################################################################################################

def conjugate_gradient_descent(A, b, x0, tol=1e-10, max_iter=10000):
    A = np.array(A)
    b = np.array(b)
    x = x0
    g = b - A.dot(x)
    d = g.copy()
    for k in range(max_iter):
        gTg = g.dot(g)
        Ad = A.dot(d)
        alpha = gTg / d.dot(Ad)
        x += alpha * d
        g -= alpha * Ad
        if np.sqrt(g.dot(g)) < tol:
            print(f"Iterations required: {k+1}")
            return x
        beta = g.dot(g) / gTg
        d = g + beta * d
    return x

########################################################
'''Question 1 Part 2'''
print("\n\nQuestion 1 Part 2\n")
########################################################

boolian = True
A, b = f1(sr_num, boolian)
A = np.matrix(A)
b = b.flatten()
x0 = np.zeros(len(b))
x = conjugate_gradient_descent(A, b, x0)
print("x*: ", x)

########################################################
'''Question 1 Part 4'''
print("\n\nQuestion 1 Part 4\n")
########################################################

boolian = False
Amn, bmn = f1(sr_num, boolian)
A = np.matrix(np.dot(Amn.T,Amn))
b = Amn.T.dot(bmn)
x0 = np.zeros(len(b))
b = b.flatten()
x = conjugate_gradient_descent(A, b, x0)
print("x*: ", x)
print("\n")


##############################################################################################################
'''Question 2'''
##############################################################################################################

def gradient_descent(x0, alpha, max_iter):
    x = x0
    xvals = [x0]
    for i in range(max_iter):
        x = x - alpha * f2(x, sr_num, 1)
        xvals.append(x)
    return xvals

def newton_method(x0, max_iter):
    x = x0
    xvals = [x0]
    for i in range(max_iter):
        x = x - f2(x, sr_num, 2)
        xvals.append(x)
    return xvals

def plot21(xvals, alphavals):
    n = len(alphavals)
    nrows = (n + 1) // 2
    fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 5 * nrows))
    axs = axs.flatten()
    for i in range(n):
        yvals = [f2(x, sr_num, 0) for x in xvals[i]]
        axs[i].plot(yvals)
        axs[i].set_xlabel(r"$Iterations$")
        axs[i].set_ylabel(r"$f_2(x)$")
        axs[i].set_title(r"$f_2(x)$ vs $Iterations$ for $\alpha = $" + str(alphavals[i]))
        axs[i].grid()
    for j in range(n, len(axs)):
        axs[j].axis('off')
    plt.subplots_adjust(left=0.075, right=0.96, bottom=0.062, top=0.946, wspace=0.443, hspace=0.46)
    plt.tight_layout()
    # plt.savefig("21.png")
    plt.show()
    
def plot22(xvals):
    xvals = np.array(xvals)
    yvals = [f2(x, sr_num, 0) for x in xvals]
    plt.plot(yvals)
    plt.xlabel(r"$Iterations$")
    plt.ylabel(r"$f_2(x)$")
    plt.title(r"$f_2(x)$ vs $Iterations$ for Newton's Method")
    plt.grid()
    # plt.savefig("22.png")
    plt.show()
    
def plot23(xvals):
    n = len(xvals)
    nrows = (n+1) // 2
    fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 2.5 * n))
    axs = axs.flatten()
    for i in range(n):
        yvals = [f2(x, sr_num, 0) for x in xvals[i]]
        axs[i].plot(yvals)
        axs[i].set_xlabel(r"$Iterations$")
        axs[i].set_ylabel(r"$f_2(x)$")
        axs[i].set_title(r"$f_2(x)$ vs $Iterations$ for $x_0$ = " + str(xvals[i][0]))
        axs[i].grid()
    for j in range(n, len(axs)):
        axs[j].axis('off')
    plt.subplots_adjust(left=0.07, right=0.967, bottom=0.065, top=0.94, wspace=0.229, hspace=0.394)
    plt.tight_layout()
    # plt.savefig("23.png")
    plt.show()
 
########################################################
'''Question 2 Part 1'''
print("\n\nQuestion 2 Part 1\n")
########################################################

x0 = np.zeros(5)
max_iter = 100
val = [0.1, 0.01, 0.001, 0.0001, 0.00001]
xall = []
for i in range(5):
    alpha = val[i]
    x = gradient_descent(x0, alpha, max_iter)
    print("alpha: ", alpha)
    print("x*: ", x[-1])
    print("f2(x*): ", f2(x[-1], sr_num, 0))
    print("\n")
    xall.append(x)
plot21(xall, val)
    
########################################################
'''Question 2 Part 2'''
print("\n\nQuestion 2 Part 2\n")
########################################################

x0 = np.zeros(5)
max_iter = 100
xn = newton_method(x0, max_iter)
print("x*: ", xn[-1])
print("f2(x*): ", f2(xn[-1], sr_num, 0))
print("\n")
plot22(xn)

########################################################
'''Question 2 Part 3'''
print("\n\nQuestion 2 Part 3\n")
########################################################

xvals = [-20, -10, 0, 10, 20]
max_iter = 100
xall = []
for x0 in xvals:
    x0 = np.array([x0, x0, x0, x0, x0])
    x = newton_method(x0, max_iter)
    print("x0: ", x0)
    print("x*: ", x[-1])
    print("f2(x*): ", f2(x[-1], sr_num, 0))
    print("\n")
    xall.append(x)
plot23(xall)


##############################################################################################################
'''Question 3'''
##############################################################################################################

def gradient_descent(x0, alpha, max_iter):
    x = x0
    xvals = [x0]
    for i in range(max_iter):
        x = x - alpha * f3(x, sr_num, 1)
        xvals.append(x)
    return xvals

def newton_method(x0, max_iter):
    x = x0
    xvals = [x0]
    for i in range(max_iter):
        x = x - f3(x, sr_num, 2)
        xvals.append(x)
    return xvals

def plot31(xvals, alpha = None):
    xvals = np.array(xvals)
    yvals = [f3(x, sr_num, 0) for x in xvals]
    plt.plot(yvals)
    plt.xlabel(r"$Iterations$")
    plt.ylabel(r"$f_3(x)$")
    if alpha is None:
        plt.title(r"$f_3(x)$ vs $Iterations$ for Newton's Method")
    else:
        plt.title(r"$f_3(x)$ vs $Iterations$ for $\alpha = $" + str(alpha))
    plt.grid()
    # plt.savefig("31.png")
    plt.show()
    
########################################################
'''Question 3 Part 1'''
print("\n\nQuestion 3 Part 1\n")
########################################################

x0 = np.ones(5)
max_iter = 100
alpha = 0.1
x = gradient_descent(x0, alpha, max_iter)
print("x*: ", x[-1])
print("f3(x*): ", f3(x[-1], sr_num, 0))

min_f3 = np.inf
for i in range(len(x)):
    if f3(x[i], sr_num, 0) < min_f3:
        min_f3 = f3(x[i], sr_num, 0)
print("Best f3(x): ", min_f3)

plot31(x, alpha = alpha)

########################################################
'''Question 3 Part 3'''
print("\n\nQuestion 3 Part 3\n")
########################################################

x0 = np.ones(5)
max_iter = 100
x = newton_method(x0, max_iter)
for i in range(10):
    print("xval ", x[i])
    print("fval ", f3(x[i], sr_num, 0))
    
########################################################
'''Question 3 Part 4'''
print("\n\nQuestion 3 Part 4\n")
########################################################

def xf(xg, xn):
    if len(xn) == 0:
        return xg[-1]
    else:
        return xn[-1]

x0 = np.ones(5)
max_iter = 100
alpha = [0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001,0.00005,0.00001]
K = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]

mink = np.inf
minalpha = np.inf
minxg = []
minxn = []
mincost = np.inf

for a in alpha:
    for k in K:
        xg = gradient_descent(x0, a, k)
        xn = newton_method(xg[-1], max_iter-k)
        if np.isnan(f3(xn[-1], sr_num, 0)):
            continue
        else:
            cost = len(xg)-1 + 25*(len(xn)-1)
            if mincost == cost:
                if f3(xn[-1], sr_num, 0) < f3(xf(minxg,minxn), sr_num, 0):
                    if xn[-1].all() == xn[0].all():
                        xn = []
                    mincost = cost
                    mink = k
                    minalpha = a
                    minxg = xg
                    minxn = xn
            
            if mincost > cost:
                if xn[-1].all() == xn[0].all():
                    xn = []
                mincost = cost
                mink = k
                minalpha = a
                minxg = xg
                minxn = xn
            
            print("alpha: ", a)
            print("k: ", k)
            print("x*: ", xn[-1] if len(xn) > 0 else xg[-1])
            print("f3(x*): ", f3(xf(xg,xn), sr_num, 0))
            print("Cost: ", cost)
            print("\n")

fg = [f3(x, sr_num, 0) for x in minxg]
fn = [f3(x, sr_num, 0) for x in minxn]

print("Best alpha: ", minalpha)
print("Best k: ", mink)
print("Best x*: ", xf(minxg,minxn))
print("Best f3(x*): ", f3(xf(minxg,minxn), sr_num, 0))
print("Best Cost: ", mincost)

plt.plot(fg, 'bo', markersize = 5, label = "Gradient Descent")
fn_x = [i + len(fg) for i in range(len(fn)-1)]
plt.plot(fn_x, fn[1:], 'rx', markersize = 5, label = "Newton's Method")
plt.legend()
plt.xlabel(r"$Iterations$")
plt.ylabel(r"$f_3(x)$")
plt.title(r"$f_3(x)$ vs $Iterations$ for Gradient Descent and Newton's Method")
plt.grid()
# plt.savefig("34.png")
plt.show()


##############################################################################################################
'''Question 4'''
##############################################################################################################

def FindAlphaExact(x):
    x0 = np.zeros(5)
    gradfx0 = f2(x0, sr_num, 1)
    px = f2(x, sr_num, 1)
    fp = f2(-px, sr_num, 0)
    num = (px.T)@px
    deno = 2*(fp + gradfx0.T@px)
    alpha = num/deno
    return alpha

def newupdate1(x0, max_iter):
    x = x0
    xvals = [x0]
    alpha = 1
    for i in range(max_iter):
        x = x - alpha*f2(x, sr_num, 1)
        xvals.append(x)
        delta = xvals[i+1] - xvals[i]
        gamma = f2(xvals[i+1], sr_num, 1) - f2(xvals[i], sr_num, 1)
        alpha = np.dot(delta, delta)/(np.dot(delta, gamma) + 1e-100)
    return xvals
        
def newupdate2(x0, max_iter):
    x = x0
    xvals = [x0]
    alpha = 1
    for i in range(max_iter):
        x = x - alpha*f2(x, sr_num, 1)
        xvals.append(x)
        delta = xvals[i+1] - xvals[i]
        gamma = f2(xvals[i+1], sr_num, 1) - f2(xvals[i], sr_num, 1)
        alpha = np.dot(delta, gamma)/(np.dot(gamma, gamma) + 1e-100)
    return xvals

def rank1update(x0, max_iter):
    x = x0
    xvals = [x0]
    G = np.eye(len(x0))
    for i in range(max_iter):
        alpha = FindAlphaExact(x)
        x = x - alpha*(G@f2(x, sr_num, 1))
        xvals.append(x)
        delta = xvals[i+1] - xvals[i]
        gamma = f2(xvals[i+1], sr_num, 1) - f2(xvals[i], sr_num, 1)
        val = delta - G@gamma
        G = G + ((np.outer(val,val))/(np.dot(val,gamma)))
    return xvals

def gradient_descent(x0, alpha, max_iter):
    x = x0
    xvals = [x0]
    for i in range(max_iter):
        x = x - alpha * f2(x, sr_num, 1)
        xvals.append(x)
    return xvals

def plotall(x11, x12, x2, x3, alp):
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(10, 10))
    x11 = np.array(x11)
    y11 = [f2(x, sr_num, 0) for x in x11]
    axs[0][0].plot(y11)
    axs[0][0].set_title(r"$f_2(x)$ vs $Iterations$ for NewUpdate1")
    axs[0][0].set_xlabel(r"$Iterations$")
    axs[0][0].set_ylabel(r"$f_2(x)$")
    axs[0][0].grid()
    
    x12 = np.array(x12)
    y12 = [f2(x, sr_num, 0) for x in x12]
    axs[0][1].plot(y12)
    axs[0][1].set_title(r"$f_2(x)$ vs $Iterations$ for NewUpdate2")
    axs[0][1].set_xlabel(r"$Iterations$")
    axs[0][1].set_ylabel(r"$f_2(x)$")
    axs[0][1].grid()
    
    x2 = np.array(x2)
    y2 = [f2(x, sr_num, 0) for x in x2]
    axs[1][0].plot(y2)
    axs[1][0].set_title(r"$f_2(x)$ vs $Iterations$ for Rank1Update")
    axs[1][0].set_xlabel(r"$Iterations$")
    axs[1][0].set_ylabel(r"$f_2(x)$")
    axs[1][0].grid()
    
    nc = 0
    nr = 2
    for i in range(len(x3)):
        x3[i] = np.array(x3[i])
        y3 = [f2(x, sr_num, 0) for x in x3[i]]    
        if i==0:
            axs[1][1].plot(y3)
            axs[1][1].set_title(r"$f_2(x)$ vs $Iterations$ for $\alpha$ = " + str(alp[i]))
            axs[1][1].set_xlabel(r"$Iterations$")
            axs[1][1].set_ylabel(r"$f_2(x)$")
            axs[1][1].grid()
        else:
            if nc == 2:
                nc = 0
                nr = nr + 1
            axs[nr][nc].plot(y3)
            axs[nr][nc].set_title(r"$f_2(x)$ vs $Iterations$ for $\alpha$ = " + str(alp[i]))
            axs[nr][nc].set_xlabel(r"$Iterations$")
            axs[nr][nc].set_ylabel(r"$f_2(x)$")
            axs[nr][nc].grid()
            nc = nc + 1
    plt.tight_layout()
    # plt.savefig("42.png")
    plt.show()

########################################################
'''Question 4 Part 2'''
print("\n\nQuestion 4 Part 2\n")
########################################################

x0 = np.zeros(5)
max_iter = 100

x11 = newupdate1(x0, max_iter)
print("x*: ", x11[-1])
print("f2(x*): ", f2(x11[-1], sr_num, 0))
print("\n")

x12 = newupdate2(x0, max_iter)
print("x*: ", x12[-1])
print("f2(x*): ", f2(x12[-1], sr_num, 0))
print("\n")

x2 = rank1update(x0, max_iter)
print("x*: ", x2[-1])
print("f2(x*): ", f2(x2[-1], sr_num, 0))
print("\n")

alp = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
x3all = []
for alpha in alp:
    x3 = gradient_descent(x0, alpha, max_iter)
    print("alpha: ", alpha)
    print("x*: ", x3[-1])
    print("f2(x*): ", f2(x3[-1], sr_num, 0))
    print("\n")
    x3all.append(x3)
    
plotall(x11, x12, x2, x3all, alp)
