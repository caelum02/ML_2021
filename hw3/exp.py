from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from q2 import *
def exp1():
    # Fix random seed
    np.random.seed(1337)

    # Generate dataset
    X_train, y = make_blobs(n_samples=500, centers=2, n_features=2, random_state=1)     
    y[y == 0] = -1

    # Set model parameters and initial values
    C = 1.0
    n = len(X_train)
    initial_alphas = np.zeros(n)
    initial_b = 0.0

    # Set tolerance and epsilon
    tol = 0.01 # error tolerance
    eps = 0.01 # alpha tolerance

    # Instantiate model
    model = SMOModel(X_train, y, C, linear_kernel,
                            initial_alphas, initial_b, np.zeros(n), tol, eps)
    model.train()
    model.plot_decision_boundary()

def exp2():
    # Fix random seed
    np.random.seed(1337)

    # Generate dataset
    X_train, y = make_moons(n_samples=500, noise=0.1, random_state=1)
    y[y == 0] = -1

    # Set model parameters and initial values
    C = 1.0
    n = len(X_train)
    initial_alphas = np.zeros(n)
    initial_b = 0.0

    # Set tolerance and epsilon
    tol = 0.01 # error tolerance
    eps = 0.01 # alpha tolerance

    # Instantiate model
    model = SMOModel(X_train, y, C, lambda x,y: gaussian_kernel(x,y, sigma=0.5),
                              initial_alphas, initial_b, np.zeros(n), tol, eps)
    
    model.train()
    model.plot_decision_boundary()

def exp3():
    # Fix random seed
    np.random.seed(1337)

    # Generate dataset
    X_train, y = make_circles(n_samples=500, noise=0.1, factor=0.1, random_state=1)
    y[y == 0] = -1

    # Set model parameters and initial values
    C = 1.0
    n = len(X_train)
    initial_alphas = np.zeros(n)
    initial_b = 0.0

    # Set tolerance and epsilon
    tol = 0.01 # error tolerance
    eps = 0.01 # alpha tolerance

    # Instantiate model
    model = SMOModel(X_train, y, C, gaussian_kernel,
                    initial_alphas, initial_b, np.zeros(n),tol,eps)
    model.train()
    model.plot_decision_boundary()  
