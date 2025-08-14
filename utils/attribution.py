import numpy as np
import pandas as pd

class AttributionAnalysis:
    """
    Attribution Analysis for Root Cause Localization as described in Section 4.2
    of the paper "BALANCE: Bayesian Linear Attribution for Root Cause Localization".
    
    This class implements various attribution methods to explain the anomalies in
    target KPIs by attributing them to candidate root causes.
    
    Parameters
    ----------
    model : object
        A fitted model with beta_mean_ attribute containing the regression coefficients.
    """
    
    def __init__(self, model):
        """
        Initialize the AttributionAnalysis with a fitted model.
        
        Parameters
        ----------
        model : object
            A fitted model with beta_mean_ attribute containing the regression coefficients.
        """
        self.model = model
        
    def sensitivity_attribution(self):
        """
        Compute attribution scores based on sensitivity (regression coefficients).
        
        Returns
        -------
        scores : array-like
            Attribution scores for each feature.
        """
        return np.abs(self.model.beta_mean_)
    
    def salience_attribution(self, X):
        """
        Compute attribution scores based on salience (gradient×input).
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            The candidate root causes.
            
        Returns
        -------
        scores : array-like
            Attribution scores for each feature.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Compute mean of each feature
        x_mean = np.mean(X, axis=0)
        
        # Compute salience
        return np.abs(self.model.beta_mean_ * x_mean)
    
    def marginal_effect_attribution(self, X, baseline=None):
        """
        Compute attribution scores based on marginal effect.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            The candidate root causes.
        baseline : array-like, optional
            Baseline values for X. If None, the mean of the first half of X is used
            (assuming the first half represents normal behavior).
            
        Returns
        -------
        scores : array-like
            Attribution scores for each feature.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Compute baseline if not provided
        if baseline is None:
            # Assume first half of the time series is normal
            n = X.shape[0]
            baseline = np.mean(X[:n//2], axis=0)
        
        # Compute delta_x (difference between anomaly and baseline)
        # For simplicity, we use the mean of the second half as the anomaly
        delta_x = np.mean(X[X.shape[0]//2:], axis=0) - baseline
        
        # Compute marginal effect
        return np.abs(self.model.beta_mean_ * delta_x)
    
    def scale_invariant_attribution(self, X, y, baseline_X=None, baseline_y=None):
        """
        Compute scale-invariant attribution scores.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            The candidate root causes.
        y : array-like
            The target KPI.
        baseline_X : array-like, optional
            Baseline values for X. If None, the mean of the first half of X is used.
        baseline_y : float, optional
            Baseline value for y. If None, the mean of the first half of y is used.
            
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
        
        # Compute baselines if not provided
        n = X.shape[0]
        if baseline_X is None:
            baseline_X = np.mean(X[:n//2], axis=0)
        
        if baseline_y is None:
            baseline_y = np.mean(y[:n//2])
        
        # Compute delta_x and delta_y
        delta_x = np.mean(X[n//2:], axis=0) - baseline_X
        delta_y = np.mean(y[n//2:]) - baseline_y
        
        # Avoid division by zero
        if abs(delta_y) < 1e-10:
            delta_y = 1e-10
        
        # Compute scale-invariant attribution
        #print(f"model.beta_mean_: {self.model.beta_mean_}")
        return np.abs(self.model.beta_mean_ * delta_x / delta_y)
    
    def direct_scale_invariant_attribution(self, X, y):
        """
        Compute direct scale-invariant attribution scores: beta * x / y
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Today's candidate root causes (single row).
        y : float or array-like
            Today's target KPI value.
            
        Returns
        -------
        scores : array-like
            Attribution scores for each feature.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Ensure X is 2D and get the single row
        if X.ndim == 1:
            X = X.reshape(1, -1)
        x_today = X[0]  # Get the single row
        
        # Ensure y is a scalar
        if isinstance(y, (list, np.ndarray)):
            y_today = y[0] if len(y) > 0 else 0
        else:
            y_today = y
        
        # Avoid division by zero
        if abs(y_today) < 1e-10:
            y_today = 1e-10
        
        # Compute direct scale-invariant attribution: beta * x / y
        scores = self.model.beta_mean_ * x_today / y_today
        
        return np.abs(scores)
    
    def get_top_k_causes(self, scores, k=5):
        """
        Get the indices of the top k root causes based on attribution scores.
        
        Parameters
        ----------
        scores : array-like
            Attribution scores for each feature.
        k : int, default=5
            Number of top causes to return.
            
        Returns
        -------
        top_k_indices : array-like
            Indices of the top k root causes.
        """
        return np.argsort(-scores)[:k]