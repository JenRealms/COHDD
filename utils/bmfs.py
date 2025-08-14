import numpy as np
import pandas as pd
from scipy.special import gamma, gammainc, erf, hyperu
from scipy.stats import beta
from sklearn.mixture import GaussianMixture
import time

class BMFS:
    """
    Bayesian Multicollinear Feature Selection (BMFS) as described in the paper
    "BALANCE: Bayesian Linear Attribution for Root Cause Localization"
    
    This class implements a Bayesian sparse linear model that can handle multicollinearity
    among features and missing values in the data.
    
    Parameters
    ----------
    max_iter : int, default=100
        Maximum number of iterations for variational inference.
    
    tol : float, default=1e-4
        Tolerance for convergence.
        
    positive : bool, default=False
        Whether to constrain the coefficients to be positive.
        
    verbose : bool, default=False
        Whether to  progress during fitting.
    """
    
    def __init__(self, max_iter=100, tol=1e-4, positive=False, verbose=False):
        self.max_iter = max_iter
        self.tol = tol
        self.positive = positive
        self.verbose = verbose
        self.beta_mean_ = None
        self.beta_cov_ = None

    @staticmethod
    def _compute_U_function(a, b, z):
    # If z is a scalar, just process it directly

            return hyperu(a, b, z)
        
    def _compute_gamma_functions(self, c, d):
        """
        Compute the upper incomplete gamma function Γ(c,d).
        
        Parameters
        ----------
        c : float or array-like
            First parameter.
        d : float or array-like
            Second parameter.
            
        Returns
        -------
        result : float or array-like
            Upper incomplete gamma function value.
        """
        # Clip d to avoid zero or negative values
        d = np.clip(d, 1e-8, None)
        # For simplicity, we use an approximation for the upper incomplete gamma function
        # In a production environment, a more accurate implementation would be needed
        #if c == -1:
        #    return np.exp(-d) / d
        #elif c == -0.5:
        #    return np.sqrt(np.pi) * np.exp(-d) * (1 - erf(np.sqrt(d)))
        #else:
        #    return gamma(c) * (1 - gammainc(c, d))
        #return self._compute_U(c, d)

        if np.isscalar(d):
            if d > 709:
                # handle overflow, e.g., clip or warn
                return 1e-10
            else:
                #print(f"d: {d}")
                return BMFS._compute_U_function(1-c, 1-c, d)/ np.exp(d)
        else:
            # d is an array, process each element
            result = np.empty_like(d, dtype=float)
            for idx, val in np.ndenumerate(d):
                if val > 709:
                    # handle overflow, e.g., clip or warn
                    result[idx] = 1e-10
                else:
                    result[idx] = BMFS._compute_U_function(1-c, 1-c, val)/ np.exp(val)
            return result
    
    def _compute_lambda_moments(self, d):
        """
        Compute the moments of λ given the variational parameter d.
        
        Parameters
        ----------
        d : array-like
            Variational parameter for λ.
            
        Returns
        -------
        lambda_mean : array-like
            Mean of λ.
        lambda_sqrt_mean : array-like
            Mean of λ^(1/2).
        """
        # <λ> = Γ(-1, d) / Γ(0, d)
        lambda_mean = self._compute_gamma_functions(-1, d) / self._compute_gamma_functions(0, d)

        #temp = self._compute_gamma_functions(0, d)
        # <λ^(1/2)> = Γ(1.5)Γ(-0.5, d) / Γ(0, d)
        lambda_sqrt_mean = gamma(1.5) * self._compute_gamma_functions(-0.5, d) / self._compute_gamma_functions(0, d)
        
        return lambda_mean, lambda_sqrt_mean
    
    def _soft_thresholding(self, beta_mean, lambda_mean):
        """
        Perform soft thresholding on beta coefficients based on lambda values.
        
        Parameters
        ----------
        beta_mean : array-like
            Mean of beta coefficients.
        lambda_mean : array-like
            Mean of lambda values.
            
        Returns
        -------
        beta_mean : array-like
            Thresholded beta coefficients.
        """
        # Compute shrinkage weights ω = <λ_j>/(1 + <λ_j>)
        omega = lambda_mean / (1 + lambda_mean)
        #print("NaNs in omega before nan_to_num:", np.isnan(omega).sum())
        omega = np.nan_to_num(omega, nan=0.0, posinf=0.0, neginf=0.0)
        #print("NaNs in omega after nan_to_num:", np.isnan(omega).sum())
        
        # Fit a two-component Gaussian mixture model
        omega_reshaped = omega.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, covariance_type='full')
        gmm.fit(omega_reshaped)
        
        # Fix the means to the smallest and largest values
        means = np.array([[np.min(omega)], [np.max(omega)]])
        gmm.means_ = means
        gmm.fit(omega_reshaped)  # Refit with fixed means
        
        # Compute the density for a range of values
        x = np.linspace(np.min(omega), np.max(omega), 1000).reshape(-1, 1)
        density = np.exp(gmm.score_samples(x))
        
        # Find the threshold as the minimum of the density
        threshold = x[np.argmin(density)][0]
        
        # Set beta_mean to zero if omega > threshold
        beta_mean[omega <= threshold] = 0
        
        return beta_mean
    
    def fit(self, X, y):
        """
        Fit the BMFS model to the data.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            The candidate root causes. Can contain missing values (NaN).
        y : array-like
            The target KPI. Can contain missing values (NaN).
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Convert pandas DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Ensure y is a column vector
        y = np.asarray(y).reshape(-1, 1)
        
        n, p = X.shape
        #print(f"n: {n}, p: {p}")
        
        # Initialize variational parameters
        if self.positive:
            # For log-normal distribution
            h_beta = np.zeros(p)
            zeta_beta = np.ones(p)
            beta_log_mean = h_beta / zeta_beta
            beta_log_var = 1 / zeta_beta
            beta_mean = np.exp(beta_log_mean + 0.5 * beta_log_var)
            beta_squared_mean = np.exp(2 * beta_log_mean + 2 * beta_log_var)
        else:
            # For Gaussian distribution
            J_beta = np.eye(p)
            h_beta = np.zeros(p)
            beta_mean = np.linalg.solve(J_beta, h_beta)
            beta_cov = np.linalg.inv(J_beta)
            beta_squared_mean = beta_mean**2 + np.diag(beta_cov)
            
  
        # Initialize other variational parameters
        d = np.ones(p)
        lambda_mean, lambda_sqrt_mean = self._compute_lambda_moments(d)
        
        #print(f"lambda_mean: {lambda_mean}, lambda_sqrt_mean: {lambda_sqrt_mean}")

        a_alpha = n / 2
        b_alpha = np.sum((y - X @ beta_mean.reshape(-1, 1))**2) / 2
        alpha_mean = a_alpha / b_alpha
        
        a_gamma = p / 2
        b_gamma = p / 2
        gamma_mean = a_gamma / b_gamma
        
        # Missing value indicators
        X_missing = np.isnan(X)
        y_missing = np.isnan(y)
        
        # Impute missing values initially
        if np.any(X_missing):
            X[X_missing] = 0  # Initial imputation
        
        if np.any(y_missing):
            y[y_missing] = np.mean(y[~y_missing])  # Initial imputation
        
        # Main variational inference loop
        for iter in range(self.max_iter):
            # Store old values for convergence check
            old_beta_mean = beta_mean.copy()
            
            # Compute X^T X and X^T y
            XTX = X.T @ X
            XTy = X.T @ y
            
            # Update q(β)
            if self.positive:
                # Update for log-normal distribution
                c1 = np.diag(alpha_mean * XTX + gamma_mean * np.diag(lambda_sqrt_mean) @ XTX @ np.diag(lambda_sqrt_mean)) * beta_squared_mean
                c2 = alpha_mean * XTy.flatten() - (alpha_mean * XTX - np.diag(np.diag(alpha_mean * XTX)) + 
                                                  gamma_mean * np.diag(lambda_sqrt_mean) @ XTX @ np.diag(lambda_sqrt_mean) - 
                                                  np.diag(np.diag(gamma_mean * np.diag(lambda_sqrt_mean) @ XTX @ np.diag(lambda_sqrt_mean)))) @ beta_mean
                
                h_beta_new = -c1 * (1 - 2 * beta_log_mean) + c2 * (1 - beta_log_mean) + 1
                zeta_beta_new = 2 * c1 - c2
                
                # Apply step size (simple approach)
                rho = 0.1
                h_beta = (1 - rho) * h_beta + rho * h_beta_new
                zeta_beta = (1 - rho) * zeta_beta + rho * zeta_beta_new
                
                # Compute mean parameters
                beta_log_mean = h_beta / zeta_beta
                beta_log_var = 1 / zeta_beta
                beta_mean = np.exp(beta_log_mean + 0.5 * beta_log_var)
                beta_squared_mean = np.exp(2 * beta_log_mean + 2 * beta_log_var)
            else:
                # Update for Gaussian distribution
                J_beta_new = alpha_mean * XTX + gamma_mean * np.diag(lambda_sqrt_mean) @ XTX @ np.diag(lambda_sqrt_mean)
                h_beta_new = alpha_mean * XTy.flatten()
                
                #if iter%10 == 0:
                #    print(f"Iteration {iter+1}: alpha_mean: {alpha_mean}, gamma_mean: {gamma_mean}")

                # Apply step size (simple approach)
                rho = 0.01
                J_beta = (1 - rho) * J_beta + rho * J_beta_new
                h_beta = (1 - rho) * h_beta + rho * h_beta_new
                
                # Compute mean parameters
                beta_cov = np.linalg.inv(J_beta)
                beta_mean = beta_cov @ h_beta
                beta_squared_mean = beta_mean**2 + np.diag(beta_cov)
                
            # Update q(λ)
            beta_beta_T = np.outer(beta_mean, beta_mean) + beta_cov if not self.positive else np.outer(beta_mean, beta_mean)
            #if iter%100 == 0:
            #    print(f"{iter}. d: {d}")
            c3 = (lambda_sqrt_mean * d - gamma(1.5) * np.sqrt(d)) / (lambda_mean * d - 1)
            
            d_new = gamma_mean * (
                np.diag(XTX * beta_beta_T) / 2 + 
                (XTX - np.diag(np.diag(XTX))) * beta_beta_T @ lambda_sqrt_mean * c3
            )

            # Apply step size
            rho = 0.001
            d = (1 - rho) * d + rho * d_new
            
            # Compute lambda moments
            lambda_mean, lambda_sqrt_mean = self._compute_lambda_moments(d)
            
            # Diagnostic print statements
            #print(f"Iteration {iter+1}: Any negative d?", np.any(d < 0))
            #print(f"Iteration {iter+1}: Any zero d?", np.any(d == 0))
            #print(f"Iteration {iter+1}: d min/max:", np.min(d), np.max(d))
            #if iter%100 == 0:
            #    print(f"Iteration {iter+1}: beta_cov: {h_beta}, NaNs in lambda_mean:", np.isnan(lambda_mean).sum())
            
            # Update q(α)
            a_alpha_new = n / 2
            b_alpha_new = (np.sum((y - X @ beta_mean.reshape(-1, 1))**2) + np.trace(beta_beta_T @ XTX) - 2 * y.T @ X @ beta_mean) / 2
            
            # Apply step size
            a_alpha = (1 - rho) * a_alpha + rho * a_alpha_new
            b_alpha = (1 - rho) * b_alpha + rho * b_alpha_new
            
            # Compute alpha mean
            alpha_mean = a_alpha / b_alpha
            
            # Update q(γ)
            a_gamma_new = p / 2
            b_gamma_new = (
                lambda_mean @ np.diag(XTX * beta_beta_T) / 2 + 
                lambda_sqrt_mean @ (XTX - np.diag(np.diag(XTX))) * beta_beta_T @ lambda_sqrt_mean / 2
            )
            
            # Apply step size
            a_gamma = (1 - rho) * a_gamma + rho * a_gamma_new
            b_gamma = (1 - rho) * b_gamma + rho * b_gamma_new
            
            # Compute gamma mean
            gamma_mean = a_gamma / b_gamma
            
            # Impute missing values if any
            if np.any(X_missing):
                for i in range(n):
                    missing_idx = np.where(X_missing[i, :])[0]
                    if len(missing_idx) > 0:
                        j = missing_idx
                        X[i, j] = np.linalg.inv(beta_beta_T[np.ix_(j, j)]) @ beta_mean[j] * y[i]
            
            if np.any(y_missing):
                for i in np.where(y_missing)[0]:
                    y[i] = X[i, :] @ beta_mean
            
            # Check convergence
            #if iter%10 == 0:
                #print(f"Iteration {iter+1}: beta_mean: {beta_mean}, old_beta_mean: {old_beta_mean}")
            #    print(f"Iteration {iter+1}: convergence measurement: {np.linalg.norm(beta_mean - old_beta_mean)}")
            #if np.max(np.abs(beta_mean - old_beta_mean)) < self.tol:
            if np.linalg.norm(beta_mean - old_beta_mean) < self.tol:
                if self.verbose:
                    print(f"Converged after {iter+1} iterations")
                break
                
            #if self.verbose and (iter + 1) % 10 == 0:
            #    print(f"Iteration {iter+1}/{self.max_iter}")
        
        # Apply soft thresholding
        #print("NaNs in lambda_mean before soft_thresholding:", np.isnan(lambda_mean).sum())
        #print("NaNs in beta_mean before soft_thresholding:", np.isnan(beta_mean).sum())
        beta_mean = self._soft_thresholding(beta_mean, lambda_mean)
        
        # Store results
        self.beta_mean_ = beta_mean
        self.beta_cov_ = beta_cov if not self.positive else None
        self.lambda_mean_ = lambda_mean
        self.alpha_mean_ = alpha_mean
        self.gamma_mean_ = gamma_mean
        
        return self
    
    def predict(self, X):
        """
        Predict using the fitted model.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            The input features.
            
        Returns
        -------
        y_pred : array-like
            The predicted values.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return X @ self.beta_mean_.reshape(-1, 1)
    
    def get_attribution_scores(self, X, y, baseline=None):
        """
        Compute attribution scores for each feature.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            The candidate root causes.
        y : array-like
            The target KPI.
        baseline : array-like, optional
            Baseline values for X. If None, the mean of X is used.
            
        Returns
        -------
        scores : array-like
            Attribution scores for each feature.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Ensure y is a column vector
        y = np.asarray(y).reshape(-1, 1)
        
        # Compute baseline if not provided
        if baseline is None:
            baseline = np.mean(X, axis=0)
            
        # Compute delta_x and delta_y
        delta_x = X - baseline
        delta_y = y - np.mean(y)
        
        # Compute attribution scores
        scores = np.abs(self.beta_mean_ * np.mean(delta_x, axis=0) / np.mean(delta_y))
        
        return scores