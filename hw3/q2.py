import numpy as np 
import matplotlib.pyplot as plt

# Linear kernel function
def linear_kernel(x, y):
    # Returns the linear combination of arrays `x` and `y` with
    # the optional bias term `b` (set to 1 by default).
    
    return x @ y.T + 1 # Note the @ operator for matrix multiplication

# Gaussian kernel function
def gaussian_kernel(x, y, sigma=1):
    # Returns the gaussian similarity of arrays `x` and `y` with
    # kernel width parameter `sigma` (set to 1 by default).
    
    if np.ndim(x) == 1 and np.ndim(y) == 1:
        result = np.exp(- (np.linalg.norm(x - y, 2)) ** 2 / (2 * sigma ** 2))
    elif (np.ndim(x) > 1 and np.ndim(y) == 1) or (np.ndim(x) == 1 and np.ndim(y) > 1):
        result = np.exp(- (np.linalg.norm(x - y, 2, axis=1) ** 2) / (2 * sigma ** 2))
    elif np.ndim(x) > 1 and np.ndim(y) > 1:
        result = np.exp(- (np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], 2, axis=2) ** 2) / (2 * sigma ** 2))
    return result

# SMO Model
class SMOModel:
    #Container object for the model used for sequential minimal optimization.
    def __init__(self, X, y, C, kernel, alphas, b, errors, tol, eps):
        self.X = X               # training data vector
        self.y = y               # class label vector
        self.C = C               # regularization parameter
        self.kernel = kernel     # kernel function
        self.alphas = alphas     # lagrange multiplier vector
        self.b = b               # scalar bias term
        self.errors = errors     # error cache
        self._obj = []           # record of objective function value
        self.n = len(self.X)     # store size of training set
        self.tol = tol           # tolerance
        self.eps = eps           # epsilon

    def get_objective(self, alphas):
        # Get objective
        return np.sum(alphas) - 0.5 * np.sum((self.y[:, None] * self.y[None, :]) * self.kernel(self.X, self.X) * (alphas[:, None] * alphas[None, :]))
    
    def support_vector_expansion(self, x):
        # Get support vector expansion of the input feature vectors in `x`.
        result = (self.alphas * self.y) @ self.kernel(self.X, x) + self.b
        return result
    
    def take_step(self, i1, i2):
        # process step for i1, i2th index pair of the training set.

        # Skip if chosen alphas are the same
        if i1 == i2:
            return 0
        
        # Get some values
        alph1 = self.alphas[i1]
        alph2 = self.alphas[i2]
        y1 = self.y[i1]
        y2 = self.y[i2]
        E1 = self.errors[i1]
        E2 = self.errors[i2]
        s = y1 * y2
        C = self.C

        # Compute kernel 
        k11 = self.kernel(self.X[i1], self.X[i1])
        k12 = self.kernel(self.X[i1], self.X[i2])
        k22 = self.kernel(self.X[i2], self.X[i2])

        #============================================
        # Fill this part
        # TODO : Compute (L) & (U), the bounds on 
        #        alph2 = alpha_2^{(t)}
        #============================================

        if y1*y2 > 0.: # y1y2 = 1
            L = np.maximum(0., alph1+alph2-C)
            U = np.minimum(C, alph1+alph2)
        else: # y1y2 = -1
            L = np.maximum(0., -alph1+alph2)
            U = np.minimum(C, -alph1+alph2+C)

        #============================================
        
        #============================================
        # Fill this part
        # TODO : Compute 2nd derivative (eta) of \ell
        #============================================

        eta = 2*k12-k11-k22

        #============================================


        # Compute new alpha 2 (a2) if eta is negative
        if (eta < 0):
            #========================================
            # Fill this part
            # TODO : Compute the solution (a2) for the
            #        case that eta is negative using
            #        alph2, y2, E1, E2, eta
            #========================================
            
            a2 = np.clip(alph2-y2*(E1-E2)/eta, L, U)



            #========================================
        # If eta is non-negative, move new a2 to bound with greater objective function value
        else:
            alphas_adj = self.alphas.copy()
            alphas_adj[i2] = L
            # objective function output with a2 = L
            Lobj = self.get_objective(alphas_adj) 
            alphas_adj[i2] = U
            # objective function output with a2 = U
            Uobj = self.get_objective(alphas_adj)
            if Lobj > (Uobj + self.eps):
                a2 = L
            elif Lobj < (Uobj - self.eps):
                a2 = U
            else:
                a2 = alph2
                
        # Push a2 to 0 or C if very close
        if a2 < 1e-8:
            a2 = 0.0
        elif a2 > (C - 1e-8):
            a2 = C
        
        # If examples can't be optimized within epsilon (eps), skip this pair
        if (np.abs(a2 - alph2) < self.eps * (a2 + alph2 + self.eps)):
            return 0
        
        #============================================
        # Fill this part
        # TODO : Calculate new alpha 1 (a1) using 
        #        a2, s, alph1, alph2
        #============================================


        a1 = alph1 + s*(alph2-a2)



        #============================================
                
        # Calculate both possible thresholds
        b1 = -E1 - y1 * (a1 - alph1) * k11 - y2 * (a2 - alph2) * k12 + self.b
        b2 = -E2 - y1 * (a1 - alph1) * k12 - y2 * (a2 - alph2) * k22 + self.b
        
        # Set new threshold based on if a1 or a2 is bound by L and/or U
        if 0 < a1 and a1 < C:
            b_new = b1
        elif 0 < a2 and a2 < C:
            b_new = b2
        # Average thresholds if both are bound
        else:
            b_new = (b1 + b2) * 0.5

        # Update model object with new alphas & threshold
        self.alphas[i1] = a1
        self.alphas[i2] = a2
        
        # Update error cache
        for index, alph in zip([i1, i2], [a1, a2]):
            if 0.0 < alph < C:
                self.errors[index] = 0.0
        
        # Set non-optimized errors based on equation 12.11 in Platt's book
        non_opt = [n for n in range(self.n) if (n != i1 and n != i2)]
        self.errors[non_opt] = self.errors[non_opt] + \
                                y1*(a1 - alph1)*self.kernel(self.X[i1], self.X[non_opt]) + \
                                y2*(a2 - alph2)*self.kernel(self.X[i2], self.X[non_opt]) - self.b + b_new
        
        # Update model threshold
        self.b = b_new
        
        return 1

    def examine_example(self, i2):
        # Examine example of i2th index training sample.

        y2 = self.y[i2]
        alph2 = self.alphas[i2]
        E2 = self.errors[i2]
        r2 = E2 * y2

        # Proceed if error is within tolerance (tol)
        if ((r2 < -self.tol and alph2 < self.C) or (r2 > self.tol and alph2 > 0)):
            
            if len(self.alphas[(self.alphas != 0) & (self.alphas != self.C)]) > 1:
                # Use 2nd choice heuristic is choose max difference in error
                if self.errors[i2] > 0:
                    i1 = np.argmin(self.errors)
                elif self.errors[i2] <= 0:
                    i1 = np.argmax(self.errors)
                step_result = self.take_step(i1, i2)
                if step_result:
                    return 1
                
            # Loop through non-zero and non-C alphas, starting at a random point
            for i1 in np.roll(np.where((self.alphas != 0) & (self.alphas != self.C))[0],
                              np.random.choice(np.arange(self.n))):
                step_result = self.take_step(i1, i2)
                if step_result:
                    return 1
            
            # loop through all alphas, starting at a random point
            for i1 in np.roll(np.arange(self.n), np.random.choice(np.arange(self.n))):
                step_result = self.take_step(i1, i2)
                if step_result:
                    return 1
        
        return 0

    def train(self):
        # Train the model

        # Initialize error cache
        self.errors = self.support_vector_expansion(self.X) - self.y

        numChanged = 0
        examineAll = 1
        while(numChanged > 0) or (examineAll):
            numChanged = 0
            if examineAll:
                # loop over all training examples
                for i in range(self.alphas.shape[0]):
                    examine_result = self.examine_example(i)
                    numChanged += examine_result
                    if examine_result:
                        obj_result = self.get_objective(self.alphas)
                        self._obj.append(obj_result)

            else:
                # loop over examples where alphas are not already at their limits
                for i in np.where((self.alphas != 0) & (self.alphas != self.C))[0]:
                    examine_result = self.examine_example(i)
                    numChanged += examine_result
                    if examine_result:
                        obj_result = self.get_objective(self.alphas)
                        self._obj.append(obj_result)
            if examineAll == 1:
                examineAll = 0
            elif numChanged == 0:
                examineAll = 1
        
    def plot_decision_boundary(self, resolution=100, colors=('b', 'k', 'r'), levels=(-1, 0, 1)):
        # Plots the decision boundary
        
        fig, ax = plt.subplots()

        xrange = np.linspace(self.X[:,0].min(), self.X[:,0].max(), resolution)
        yrange = np.linspace(self.X[:,1].min(), self.X[:,1].max(), resolution)
        grid = [[self.support_vector_expansion(np.array([xr,yr])) for xr in xrange] for yr in yrange]
        grid = np.array(grid).reshape(len(xrange), len(yrange))
        
        ax.contour(xrange, yrange, grid, levels=levels, linewidths=(1, 1, 1),
                   linestyles=('--', '-', '--'), colors=colors)
        ax.scatter(self.X[:,0], self.X[:,1],
                   c=self.y, cmap=plt.cm.viridis, lw=0, alpha=0.25)
        
        mask = np.round(self.alphas, decimals=2) != 0.0
        ax.scatter(self.X[mask,0], self.X[mask,1],
                   c=self.y[mask], cmap=plt.cm.viridis, lw=1, edgecolors='k')