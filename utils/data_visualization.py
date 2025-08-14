"""
Data Visualization Module for Root Cause Analysis

This module provides comprehensive data visualization capabilities to replace
the Forward Prediction Model functionality in the root cause attribution analysis.
It offers various visualization methods for exploring data patterns, trends,
and relationships that are crucial for understanding root causes.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from typing import Optional, List, Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings('ignore')

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

class DataVisualization:
    """
    Comprehensive data visualization for root cause analysis.
    
    This class provides various visualization methods to explore data patterns,
    trends, and relationships that help in understanding root causes without
    relying on forward prediction models.
    """
    
    def __init__(self, station_code: str, target_date: str):
        """
        Initialize the data visualization module.
        
        Parameters
        ----------
        station_code : str
            The station code for analysis.
        target_date : str
            The target date for analysis.
        """
        self.station_code = station_code
        self.target_date = target_date
        data_file_path = os.path.join(PROJECT_ROOT, 'data', 'coh_train_data.csv')
        self.raw_hist = pd.read_csv(data_file_path).fillna(0)
        # Convert ofd_date to datetime to avoid matplotlib categorical units warning
        self.raw_hist['ofd_date'] = pd.to_datetime(self.raw_hist['ofd_date'])
        
        # Clean infinity values - replace with large finite values
        numeric_cols = self.raw_hist.select_dtypes(include=[np.number])
        for col in numeric_cols.columns:
            inf_mask = np.isinf(numeric_cols[col])
            if inf_mask.any():
                # Replace infinity with a large finite value (e.g., 1e6)
                self.raw_hist.loc[inf_mask, col] = 1e6
                print(f"Warning: Replaced {inf_mask.sum()} infinity values in column '{col}' with 1e6")
        
        self.raw_hist = self.raw_hist.set_index('ofd_date')
        
        # Filter data for the specific station
        self.station_data = self.raw_hist[self.raw_hist['station_code'] == self.station_code]
        self.X_hist = self.station_data.iloc[:, 2:]  # Features
        self.y_hist = self.station_data.iloc[:, 1]   # Target (COH)
        
        # Get today's data
        from .get_station_data import GetStationData
        get_station_data = GetStationData()
        self.raw_today = get_station_data.get_station_data(self.station_code, self.target_date)
        self.X_today = self.raw_today.iloc[0, 3:]
        self.y_today = self.raw_today.iloc[0, 2]
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_time_series_overview(self, days: int = 30) -> plt.Figure:
        """
        Create a comprehensive time series overview of the station data.
        
        Parameters
        ----------
        days : int, default=30
            Number of days to show in the overview.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plots.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Time Series Overview - Station {self.station_code}', fontsize=16)
        
        # Get recent data
        recent_data = self.station_data.tail(days)
        recent_X = recent_data.iloc[:, 2:]
        recent_y = recent_data.iloc[:, 1]
        
        # Plot 1: Target KPI (COH) over time
        axes[0, 0].plot(recent_y.index, recent_y.values, 'b-', linewidth=2, label='Historical COH')
        axes[0, 0].axhline(y=recent_y.mean(), color='r', linestyle='--', alpha=0.7, label='Mean COH')
        axes[0, 0].set_title('Capacity Out Hours (COH) Trend')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('COH')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Feature correlation heatmap
        correlation_matrix = recent_X.corr()
        im = axes[0, 1].imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        axes[0, 1].set_title('Feature Correlation Matrix')
        axes[0, 1].set_xticks(range(len(correlation_matrix.columns)))
        axes[0, 1].set_yticks(range(len(correlation_matrix.columns)))
        axes[0, 1].set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
        axes[0, 1].set_yticklabels(correlation_matrix.columns)
        plt.colorbar(im, ax=axes[0, 1])
        
        # Plot 3: Top features by variance
        feature_variance = recent_X.var().sort_values(ascending=False)
        top_features = feature_variance.head(10)
        axes[1, 0].barh(range(len(top_features)), top_features.values)
        axes[1, 0].set_yticks(range(len(top_features)))
        axes[1, 0].set_yticklabels(top_features.index)
        axes[1, 0].set_title('Top 10 Features by Variance')
        axes[1, 0].set_xlabel('Variance')
        
        # Plot 4: Feature distribution comparison
        # Select top 5 features by correlation with target
        target_correlations = recent_X.corrwith(recent_y).abs().sort_values(ascending=False)
        top_corr_features = target_correlations.head(5).index
        
        for i, feature in enumerate(top_corr_features):
            axes[1, 1].hist(recent_X[feature], alpha=0.6, label=feature, bins=20)
        
        axes[1, 1].set_title('Distribution of Top Correlated Features')
        axes[1, 1].set_xlabel('Feature Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        plt.tight_layout()
        return fig
    
    def plot_anomaly_detection(self, threshold_percentile: float = 95) -> plt.Figure:
        """
        Visualize anomaly detection in the data.
        
        Parameters
        ----------
        threshold_percentile : float, default=95
            Percentile threshold for anomaly detection.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plots.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Anomaly Detection - Station {self.station_code}', fontsize=16)
        
        # Calculate thresholds
        target_threshold = np.percentile(self.y_hist, threshold_percentile)
        
        # Plot 1: Target KPI with anomaly threshold
        axes[0, 0].plot(self.y_hist.index, self.y_hist.values, 'b-', alpha=0.7, label='COH')
        axes[0, 0].axhline(y=target_threshold, color='r', linestyle='--', 
                           label=f'{threshold_percentile}th percentile')
        
        # Highlight anomalies
        anomalies = self.y_hist[self.y_hist > target_threshold]
        axes[0, 0].scatter(anomalies.index, anomalies.values, color='red', 
                           s=50, zorder=5, label='Anomalies')
        
        axes[0, 0].set_title('COH with Anomaly Detection')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('COH')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Feature-wise anomaly detection
        feature_anomalies = {}
        for feature in self.X_hist.columns:
            feature_threshold = np.percentile(self.X_hist[feature], threshold_percentile)
            anomalies_count = (self.X_hist[feature] > feature_threshold).sum()
            feature_anomalies[feature] = anomalies_count
        
        # Sort by anomaly count
        sorted_anomalies = dict(sorted(feature_anomalies.items(), 
                                     key=lambda x: x[1], reverse=True)[:10])
        
        axes[0, 1].barh(range(len(sorted_anomalies)), list(sorted_anomalies.values()))
        axes[0, 1].set_yticks(range(len(sorted_anomalies)))
        axes[0, 1].set_yticklabels(sorted_anomalies.keys())
        axes[0, 1].set_title('Feature Anomaly Count')
        axes[0, 1].set_xlabel('Number of Anomalies')
        
        # Plot 3: Correlation with anomalies
        # Create binary anomaly indicator
        anomaly_indicator = (self.y_hist > target_threshold).astype(int)
        
        # Calculate correlation with anomalies
        anomaly_correlations = self.X_hist.corrwith(anomaly_indicator).abs().sort_values(ascending=False)
        top_anomaly_features = anomaly_correlations.head(10)
        
        axes[1, 0].barh(range(len(top_anomaly_features)), top_anomaly_features.values)
        axes[1, 0].set_yticks(range(len(top_anomaly_features)))
        axes[1, 0].set_yticklabels(top_anomaly_features.index)
        axes[1, 0].set_title('Feature Correlation with Anomalies')
        axes[1, 0].set_xlabel('Correlation Coefficient')
        
        # Plot 4: Today's values vs historical distribution
        today_values = self.X_today
        feature_names = today_values.index[:10]  # Top 10 features
        
        for i, feature in enumerate(feature_names):
            historical_data = self.X_hist[feature]
            today_value = today_values[feature]
            
            # Calculate percentile of today's value
            percentile = (historical_data < today_value).mean() * 100
            
            axes[1, 1].barh(i, percentile, color='skyblue', alpha=0.7)
            axes[1, 1].text(percentile + 2, i, f'{percentile:.1f}%', 
                           va='center', fontsize=8)
        
        axes[1, 1].set_yticks(range(len(feature_names)))
        axes[1, 1].set_yticklabels(feature_names)
        axes[1, 1].set_title("Today's Values (Percentile of Historical)")
        axes[1, 1].set_xlabel('Percentile')
        axes[1, 1].set_xlim(0, 100)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance_analysis(self) -> plt.Figure:
        """
        Analyze and visualize feature importance using various statistical methods.
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plots.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Feature Importance Analysis - Station {self.station_code}', fontsize=16)
        
        # Method 1: Correlation-based importance
        correlations = self.X_hist.corrwith(self.y_hist).abs().sort_values(ascending=False)
        top_corr_features = correlations.head(15)
        
        axes[0, 0].barh(range(len(top_corr_features)), top_corr_features.values)
        axes[0, 0].set_yticks(range(len(top_corr_features)))
        axes[0, 0].set_yticklabels(top_corr_features.index)
        axes[0, 0].set_title('Feature Importance (Correlation)')
        axes[0, 0].set_xlabel('Absolute Correlation')
        
        # Method 2: Variance-based importance
        feature_variance = self.X_hist.var().sort_values(ascending=False)
        top_variance_features = feature_variance.head(15)
        
        axes[0, 1].barh(range(len(top_variance_features)), top_variance_features.values)
        axes[0, 1].set_yticks(range(len(top_variance_features)))
        axes[0, 1].set_yticklabels(top_variance_features.index)
        axes[0, 1].set_title('Feature Importance (Variance)')
        axes[0, 1].set_xlabel('Variance')
        
        # Method 3: Mutual information approximation
        # Using correlation with target as proxy for mutual information
        target_correlations = self.X_hist.corrwith(self.y_hist).abs()
        top_mi_features = target_correlations.sort_values(ascending=False).head(15)
        
        axes[1, 0].barh(range(len(top_mi_features)), top_mi_features.values)
        axes[1, 0].set_yticks(range(len(top_mi_features)))
        axes[1, 0].set_yticklabels(top_mi_features.index)
        axes[1, 0].set_title('Feature Importance (Target Correlation)')
        axes[1, 0].set_xlabel('Correlation with Target')
        
        # Method 4: Combined importance score
        # Combine correlation and variance
        combined_score = (correlations * feature_variance).sort_values(ascending=False)
        top_combined_features = combined_score.head(15)
        
        axes[1, 1].barh(range(len(top_combined_features)), top_combined_features.values)
        axes[1, 1].set_yticks(range(len(top_combined_features)))
        axes[1, 1].set_yticklabels(top_combined_features.index)
        axes[1, 1].set_title('Feature Importance (Combined Score)')
        axes[1, 1].set_xlabel('Combined Importance Score')
        
        plt.tight_layout()
        return fig
    
    def plot_temporal_patterns(self, window_size: int = 7) -> plt.Figure:
        """
        Analyze temporal patterns in the data.
        
        Parameters
        ----------
        window_size : int, default=7
            Size of the rolling window for pattern analysis.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plots.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Temporal Pattern Analysis - Station {self.station_code}', fontsize=16)
        
        # Plot 1: Rolling statistics of target
        rolling_mean = self.y_hist.rolling(window=window_size).mean()
        rolling_std = self.y_hist.rolling(window=window_size).std()
        
        axes[0, 0].plot(self.y_hist.index, self.y_hist.values, 'b-', alpha=0.6, label='COH')
        axes[0, 0].plot(rolling_mean.index, rolling_mean.values, 'r-', linewidth=2, label=f'{window_size}-day rolling mean')
        axes[0, 0].fill_between(rolling_mean.index, 
                               rolling_mean.values - rolling_std.values,
                               rolling_mean.values + rolling_std.values,
                               alpha=0.3, color='red', label='±1 std')
        axes[0, 0].set_title('Target KPI with Rolling Statistics')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('COH')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Feature trends over time
        # Select top 5 features by correlation
        top_features = self.X_hist.corrwith(self.y_hist).abs().sort_values(ascending=False).head(5).index
        
        for feature in top_features:
            feature_data = self.X_hist[feature]
            rolling_mean_feature = feature_data.rolling(window=window_size).mean()
            axes[0, 1].plot(rolling_mean_feature.index, rolling_mean_feature.values, 
                           label=feature, linewidth=2)
        
        axes[0, 1].set_title('Top Feature Trends (Rolling Mean)')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Feature Value')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Seasonal patterns (if enough data)
        if len(self.y_hist) >= 30:
            # Weekly patterns
            weekly_avg = self.y_hist.groupby(self.y_hist.index.dayofweek).mean()
            axes[1, 0].bar(range(7), weekly_avg.values, color='skyblue')
            axes[1, 0].set_title('Weekly Pattern (Average COH by Day of Week)')
            axes[1, 0].set_xlabel('Day of Week')
            axes[1, 0].set_ylabel('Average COH')
            axes[1, 0].set_xticks(range(7))
            axes[1, 0].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        else:
            # If not enough data, show distribution
            axes[1, 0].hist(self.y_hist.values, bins=20, alpha=0.7, color='skyblue')
            axes[1, 0].set_title('Target KPI Distribution')
            axes[1, 0].set_xlabel('COH')
            axes[1, 0].set_ylabel('Frequency')
        
        # Plot 4: Feature correlation evolution
        # Calculate rolling correlation for top features
        if len(self.y_hist) >= window_size:
            rolling_corr = {}
            for feature in top_features[:3]:  # Top 3 features
                feature_data = self.X_hist[feature]
                rolling_corr[feature] = feature_data.rolling(window=window_size).corr(self.y_hist)
            
            for feature, corr_data in rolling_corr.items():
                axes[1, 1].plot(corr_data.index, corr_data.values, label=feature, linewidth=2)
            
            axes[1, 1].set_title('Rolling Correlation with Target')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('Correlation Coefficient')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # If not enough data, show feature distributions
            for feature in top_features[:3]:
                axes[1, 1].hist(self.X_hist[feature], alpha=0.6, label=feature, bins=15)
            axes[1, 1].set_title('Top Feature Distributions')
            axes[1, 1].set_xlabel('Feature Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
        
        plt.tight_layout()
        return fig
    
    def plot_dimensionality_reduction(self, method: str = 'pca') -> plt.Figure:
        """
        Visualize data using dimensionality reduction techniques.
        
        Parameters
        ----------
        method : str, default='pca'
            Dimensionality reduction method ('pca' or 'tsne').
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plots.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Dimensionality Reduction Analysis - Station {self.station_code}', fontsize=16)
        
        # Prepare data
        X_scaled = StandardScaler().fit_transform(self.X_hist)
        
        if method == 'pca':
            # PCA analysis
            pca = PCA()
            X_pca = pca.fit_transform(X_scaled)
            
            # Plot explained variance
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            axes[0].plot(range(1, len(explained_variance_ratio) + 1), 
                        cumulative_variance, 'bo-', linewidth=2)
            axes[0].set_title('PCA Explained Variance')
            axes[0].set_xlabel('Number of Components')
            axes[0].set_ylabel('Cumulative Explained Variance')
            axes[0].grid(True, alpha=0.3)
            
            # Plot first two components
            scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], 
                                    c=self.y_hist.values, cmap='viridis', alpha=0.7)
            axes[1].set_title('PCA: First Two Components')
            axes[1].set_xlabel('First Principal Component')
            axes[1].set_ylabel('Second Principal Component')
            plt.colorbar(scatter, ax=axes[1], label='COH')
            
        elif method == 'tsne':
            # t-SNE analysis
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled)-1))
            X_tsne = tsne.fit_transform(X_scaled)
            
            # Plot explained variance (not applicable for t-SNE, so show feature importance)
            feature_importance = np.abs(self.X_hist.corrwith(self.y_hist)).sort_values(ascending=False)
            top_features = feature_importance.head(10)
            
            axes[0].barh(range(len(top_features)), top_features.values)
            axes[0].set_yticks(range(len(top_features)))
            axes[0].set_yticklabels(top_features.index)
            axes[0].set_title('Top Feature Importance')
            axes[0].set_xlabel('Absolute Correlation with Target')
            
            # Plot t-SNE results
            scatter = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], 
                                    c=self.y_hist.values, cmap='viridis', alpha=0.7)
            axes[1].set_title('t-SNE: 2D Projection')
            axes[1].set_xlabel('First t-SNE Component')
            axes[1].set_ylabel('Second t-SNE Component')
            plt.colorbar(scatter, ax=axes[1], label='COH')
        
        plt.tight_layout()
        return fig
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive data visualization report.
        
        Returns
        -------
        report : dict
            Dictionary containing all visualization figures and analysis results.
        """
        report = {
            'station_code': self.station_code,
            'target_date': self.target_date,
            'figures': {},
            'analysis_results': {}
        }
        
        # Generate all visualizations
        report['figures']['time_series_overview'] = self.plot_time_series_overview()
        report['figures']['anomaly_detection'] = self.plot_anomaly_detection()
        report['figures']['feature_importance'] = self.plot_feature_importance_analysis()
        report['figures']['temporal_patterns'] = self.plot_temporal_patterns()
        report['figures']['dimensionality_reduction'] = self.plot_dimensionality_reduction()
        
        # Generate analysis results
        report['analysis_results'] = {
            'top_correlated_features': self.X_hist.corrwith(self.y_hist).abs().sort_values(ascending=False).head(10).to_dict(),
            'feature_variance': self.X_hist.var().sort_values(ascending=False).head(10).to_dict(),
            'anomaly_threshold': np.percentile(self.y_hist, 95),
            'data_summary': {
                'total_samples': len(self.X_hist),
                'total_features': len(self.X_hist.columns),
                'date_range': f"{self.X_hist.index.min()} to {self.X_hist.index.max()}",
                'target_mean': self.y_hist.mean(),
                'target_std': self.y_hist.std()
            }
        }
        
        return report
    
    def save_all_visualizations(self, output_dir: str = "Output/Figures") -> None:
        """
        Save all visualization figures to files.
        
        Parameters
        ----------
        output_dir : str, default="Output/Figures"
            Directory to save the figures.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate and save all figures
        figures = {
            'time_series_overview': self.plot_time_series_overview(),
            'anomaly_detection': self.plot_anomaly_detection(),
            'feature_importance': self.plot_feature_importance_analysis(),
            'temporal_patterns': self.plot_temporal_patterns(),
            'dimensionality_reduction_pca': self.plot_dimensionality_reduction('pca'),
            'dimensionality_reduction_tsne': self.plot_dimensionality_reduction('tsne')
        }
        
        for name, fig in figures.items():
            filepath = os.path.join(output_dir, f"{self.station_code}_{name}.png")
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved {name} to {filepath}")
    
    def get_visualization_summary(self) -> str:
        """
        Generate a text summary of the visualization analysis.
        
        Returns
        -------
        summary : str
            Text summary of the analysis results.
        """
        # Calculate key metrics
        top_correlations = self.X_hist.corrwith(self.y_hist).abs().sort_values(ascending=False).head(5)
        top_variance = self.X_hist.var().sort_values(ascending=False).head(5)
        anomaly_threshold = np.percentile(self.y_hist, 95)
        anomaly_count = (self.y_hist > anomaly_threshold).sum()
        
        summary = f"""
Data Visualization Summary for Station {self.station_code}

Key Metrics:
- Total samples: {len(self.X_hist)}
- Total features: {len(self.X_hist.columns)}
- Date range: {self.X_hist.index.min()} to {self.X_hist.index.max()}
- Target (COH) mean: {self.y_hist.mean():.2f}
- Target (COH) std: {self.y_hist.std():.2f}
- Anomaly threshold (95th percentile): {anomaly_threshold:.2f}
- Number of anomalies: {anomaly_count}

Top 5 Features by Correlation with Target:
{chr(10).join([f"- {feature}: {corr:.4f}" for feature, corr in top_correlations.items()])}

Top 5 Features by Variance:
{chr(10).join([f"- {feature}: {var:.4f}" for feature, var in top_variance.items()])}

Visualization Analysis:
- Time series overview shows temporal patterns and trends
- Anomaly detection identifies unusual data points
- Feature importance analysis reveals key drivers
- Temporal patterns show seasonal and cyclical behavior
- Dimensionality reduction provides data structure insights
"""
        
        return summary