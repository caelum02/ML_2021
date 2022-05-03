import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.linalg import solve


def plot_approximation(gpr, X, Y, X_sample, Y_sample, X_next=None, show_legend=False):
    mu, std = gpr.predict(X, return_std=True)
    plt.fill_between(X.ravel(), 
                     mu.ravel() + 1.96 * std, 
                     mu.ravel() - 1.96 * std, 
                     alpha=0.1) 
    plt.plot(X, Y, 'y--', lw=1, label='Noise-free objective')
    plt.plot(X, mu, 'b-', lw=1, label='Surrogate function')
    plt.plot(X_sample, Y_sample, 'kx', mew=3, label='Noisy samples')
    if X_next:
        plt.axvline(x=X_next, ls='--', c='k', lw=1)
    if show_legend:
        plt.legend()

def plot_acquisition(X, Y, X_next, show_legend=False):
    plt.plot(X, Y, 'r-', lw=1, label='Acquisition function')
    plt.axvline(x=X_next, ls='--', c='k', lw=1, label='Next sampling location')
    if show_legend:
        plt.legend()    


def probability_improvement(X, X_sample, Y_sample, gpr, xi=0.01, eta=5):
    '''Calculate expected improvement'''    

    mu, sigma = gpr.predict(X, return_std=True)
    mu_samp, sigma_sample = gpr.predict(X_sample, return_std=True)

    idx = np.argmax(mu_samp)
    fn = mu_samp[idx]

    with np.errstate(divide='warn'):
        imp = mu.squeeze() - fn - xi
        z = -imp / sigma
        pi = norm.cdf(-z).reshape(-1)

    return pi + eta * sigma / np.sqrt(X.shape[0])


def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25):
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None

    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr)

    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x

    return min_x.reshape(-1, 1)


if __name__ == '__main__':
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern

    noise = 0.2

    def f(X, noise=noise):
        return -np.cos(3*X) - 1.2*X**2 + 0.5*X + noise * np.random.randn(*X.shape)


    # Acquisition function
    acquisition_fn = probability_improvement

    X_init = np.array([-1.0, 0.0, 1.5]).reshape(-1, 1)
    Y_init = f(X_init)
    bounds = np.array([[-1.0, 2.0]])
    X = np.arange(bounds[:, 0], bounds[:, 1], 0.01).reshape(-1, 1)
    Y = f(X,0)

    # Gaussian process with Matern kernel as surrogate model
    m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gpr = GaussianProcessRegressor(kernel=m52, alpha=noise**2)

    # Initialize samples
    X_sample = X_init
    Y_sample = Y_init

    n_iter = 10
    plt.figure(figsize=(12, n_iter * 3))
    plt.subplots_adjust(hspace=0.4)
    for i in range(n_iter):
        gpr.fit(X_sample, Y_sample)

        # Obtain next sampling point from the acquisition function (expected_improvement)
        X_next = propose_location(acquisition_fn, X_sample, Y_sample, gpr, bounds)    
        Y_next = f(X_next, noise)
        
        # Plot samples, surrogate function, noise-free objective and next sampling location
        plt.subplot(n_iter, 2, 2 * i + 1)
        plot_approximation(gpr, X, Y, X_sample, Y_sample, X_next, show_legend=i==0)
        plt.title(f'Iteration {i+1}')
        plt.subplot(n_iter, 2, 2 * i + 2)
        plot_acquisition(X, acquisition_fn(X, X_sample, Y_sample, gpr), X_next, show_legend=i==0)
        
        # Add sample to previous samples
        X_sample = np.vstack((X_sample, X_next))
        Y_sample = np.vstack((Y_sample, Y_next))

    plt.savefig('./result.png')
