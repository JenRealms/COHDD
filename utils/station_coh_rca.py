import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, Any

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# Import the BMFS and AttributionAnalysis classes
from .bmfs import BMFS
from .attribution import AttributionAnalysis
from .get_station_data import GetStationData #This line!!!
from .data_visualization import DataVisualization
#from .input_sql_query import get_station_sql_query

class StationCOHRCA:
    """
    Root Cause Analysis for Station Cappped Out Hours
    
    This class uses data visualization and attribution analysis to identify root causes
    for station capacity overages in a last mile organization.
    """
    
    def __init__(self, station_code, target_date, positive=False, max_iter=2500, verbose=False):
        """
        Initialize the RCA system.
        
        Parameters
        ----------
        station_code : str
            The station code for analysis.
        target_date : str
            The target date for analysis.
        positive : bool, default=False
            Whether to constrain coefficients to be positive.
        max_iter : int, default=2500
            Maximum number of iterations for BMFS.
        verbose : bool, default=False
            Whether to print progress during fitting.
        """
        self.station_code = station_code
        self.target_date = target_date
        self.positive = positive
        self.max_iter = max_iter
        self.verbose = verbose
        
        # Initialize data visualization module (replaces Forward Prediction Model)
        self.data_viz = DataVisualization(station_code, target_date)
        
        # Initialize BMFS model for attribution analysis
        self.model = BMFS(max_iter=3000, tol=1e-3)
        self.attribution = None
        self.feature_names = None
        self.fitted = False
        
        # Load data
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
                #print(f"Warning: Replaced {inf_mask.sum()} infinity values in column '{col}' with 1e6")
        
        self.raw_hist = self.raw_hist.set_index('ofd_date')
        self.X_hist = self.raw_hist[self.raw_hist['station_code'] == self.station_code].iloc[:, 2:]
        self.y_hist = self.raw_hist[self.raw_hist['station_code'] == self.station_code].iloc[:, 1]
        
        # Get today's data
        self.get_station_data = GetStationData()
        self.raw_today = self.get_station_data.get_station_data(self.station_code, self.target_date)
        self.X_today = self.raw_today.iloc[0, 3:]
        self.y_today = self.raw_today.iloc[0, 2]

    def fit(self):
        """
        Fit the model using historical data and generate data visualizations.
        
        Returns
        -------
        self : object
            Returns self.
        """
        # Store feature names
        self.feature_names = self.X_hist.columns if isinstance(self.X_hist, pd.DataFrame) else None
        
        # Fit the BMFS model for attribution analysis
        self.model.fit(self.X_hist, self.y_hist)
        
        # Create attribution analyzer
        self.attribution = AttributionAnalysis(self.model)
        
        self.fitted = True
        return self
    
    def generate_data_visualizations(self) -> Dict[str, Any]:
        """
        Generate comprehensive data visualizations to replace Forward Prediction Model.
        
        Returns
        -------
        viz_report : dict
            Dictionary containing all visualization figures and analysis results.
        """
        if not self.fitted:
            self.fit()
        
        return self.data_viz.generate_comprehensive_report()
    
    def save_visualizations(self, output_dir: str = "Output/Figures") -> None:
        """
        Save all data visualization figures to files.
        
        Parameters
        ----------
        output_dir : str, default="Output/Figures"
            Directory to save the figures.
        """
        self.data_viz.save_all_visualizations(output_dir)
    
    def get_visualization_summary(self) -> str:
        """
        Get a text summary of the data visualization analysis.
        
        Returns
        -------
        summary : str
            Text summary of the visualization analysis.
        """
        return self.data_viz.get_visualization_summary()

    def analyze_root_causes(self, method='scale_invariant', k=None, baseline_X=None, baseline_y=None):
        """
        Analyze root causes for today's capacity overage using data visualization and attribution.
        
        Parameters
        ----------
        method : str, default='scale_invariant'
            Attribution method to use. Options: 'sensitivity', 'salience', 
            'marginal_effect', 'scale_invariant', 'direct_scale_invariant'.
        k : int, default=3
            Number of top root causes to return.
        baseline_X : pandas.DataFrame or array-like, optional
            Baseline values for X. If None, uses historical mean.
        baseline_y : float, optional
            Baseline value for y. If None, uses historical mean.
            
        Returns
        -------
        result : dict
            Dictionary containing analysis results including visualization insights.
        """
        if not self.fitted:
            self.fit()
        
        # Generate data visualizations (replaces Forward Prediction Model)
        #viz_report = self.generate_data_visualizations()
        
        # Perform attribution analysis
        if method == 'sensitivity':
            scores = self.attribution.sensitivity_attribution()
        elif method == 'salience':
            scores = self.attribution.salience_attribution(self.X_hist)
        elif method == 'marginal_effect':
            scores = self.attribution.marginal_effect_attribution(self.X_hist, baseline_X)
        elif method == 'scale_invariant':
            scores = self.attribution.scale_invariant_attribution(self.X_hist, self.y_hist, baseline_X, baseline_y)
        elif method == 'direct_scale_invariant':
            scores = self.attribution.direct_scale_invariant_attribution(self.X_hist, self.y_hist)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Get top k root causes
        if k is None:
            k = 15
        
        top_indices = np.argsort(-scores)[:k]
        top_scores = scores[top_indices]
        
        # Get feature names
        feature_names = self.feature_names if self.feature_names is not None else [f"Feature {i}" for i in range(len(scores))]
        top_features = [feature_names[i] for i in top_indices]
        
        # Combine visualization insights with attribution results
        result = {
            'method': method,
            'all_scores': scores,
            'top_indices': top_indices,
            'top_scores': top_scores,
            'top_features': top_features,
            #'visualization_report': viz_report,
            #'data_summary': viz_report['analysis_results']['data_summary'],
            #'anomaly_threshold': viz_report['analysis_results']['anomaly_threshold'],
            #'top_correlated_features': viz_report['analysis_results']['top_correlated_features']
        }
        
        return result

    def visualize_results(self, result, show_history=True, history_days=30):
        """
        Visualize the RCA results with enhanced data visualization.
        
        Parameters
        ----------
        result : dict
            Result from analyze_root_causes method.
        show_history : bool, default=True
            Whether to show historical trends.
        history_days : int, default=30
            Number of days of history to show.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        # Create a comprehensive visualization combining attribution and data insights
        fig = plt.figure(figsize=(20, 15))
        
        # Plot 1: KPI trend with anomaly detection
        ax1 = plt.subplot(3, 3, 1)
        
        if show_history:
            # Limit history to specified days
            if isinstance(self.y_hist, pd.Series) and isinstance(self.y_hist.index, pd.DatetimeIndex):
                recent_y = self.y_hist.iloc[-history_days:]
                ax1.plot(recent_y.index, recent_y, 'b-', label='Historical COH')
                
                # Add anomaly threshold
                anomaly_threshold = result.get('anomaly_threshold', np.percentile(self.y_hist, 95))
                ax1.axhline(y=anomaly_threshold, color='r', linestyle='--', alpha=0.7, label='Anomaly Threshold')
                
                # Add today's point
                if isinstance(self.y_today, pd.Series):
                    ax1.plot(self.y_today.index, self.y_today, 'ro', label='Today')
                else:
                    ax1.plot(recent_y.index[-1] + pd.Timedelta(days=1), self.y_today, 'ro', label='Today')
            else:
                recent_y = self.y_hist[-history_days:]
                ax1.plot(range(-history_days, 0), recent_y, 'b-', label='Historical COH')
                ax1.plot(0, self.y_today, 'ro', label='Today')
                
        ax1.set_title('Capacity Overage KPI Trend')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('COH')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Attribution scores
        ax2 = plt.subplot(3, 3, 2)
        
        # Get feature names
        feature_names = self.feature_names if self.feature_names is not None else [f"Feature {i}" for i in range(len(result['all_scores']))]
        
        # Sort scores for better visualization
        sorted_indices = np.argsort(-result['all_scores'])
        sorted_scores = result['all_scores'][sorted_indices]
        sorted_features = [feature_names[i] for i in sorted_indices]
        
        # Limit to top 15 for readability
        top_n = min(15, len(sorted_scores))
        ax2.barh(range(top_n), sorted_scores[:top_n], color='skyblue')
        ax2.set_yticks(range(top_n))
        ax2.set_yticklabels(sorted_features[:top_n])
        ax2.set_title(f'Attribution Scores ({result["method"]})')
        ax2.set_xlabel('Attribution Score')
        
        # Plot 3: Top root causes over time
        if show_history and isinstance(self.X_hist, pd.DataFrame):
            ax3 = plt.subplot(3, 3, 3)
            
            # Get recent history
            if isinstance(self.X_hist.index, pd.DatetimeIndex):
                recent_X = self.X_hist.iloc[-history_days:]
            else:
                recent_X = self.X_hist.iloc[-history_days:]
            
            # Plot top root causes
            for i, idx in enumerate(result['top_indices'][:5]):  # Limit to top 5
                feature_name = feature_names[idx]
                if isinstance(recent_X.index, pd.DatetimeIndex):
                    ax3.plot(recent_X.index, recent_X[feature_name], 
                            label=f"{feature_name} (Score: {result['top_scores'][i]:.4f})")
                    
                    # Add today's point
                    if isinstance(self.X_today, pd.DataFrame) and feature_name in self.X_today.columns:
                        ax3.plot(self.X_today.index, self.X_today[feature_name], 'o', 
                                color=f'C{i}', markersize=8)
                else:
                    ax3.plot(range(-history_days, 0), recent_X[feature_name], 
                            label=f"{feature_name} (Score: {result['top_scores'][i]:.4f})")
                    
                    # Add today's point
                    if isinstance(self.X_today, pd.DataFrame) and feature_name in self.X_today.columns:
                        ax3.plot(0, self.X_today[feature_name].values[0], 'o', 
                                color=f'C{i}', markersize=8)
            
            ax3.set_title('Top Root Causes Over Time')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Value')
            ax3.legend()
            ax3.grid(True)
        
        # Plot 4: Feature correlation with target
        ax4 = plt.subplot(3, 3, 4)
        top_corr_features = result.get('top_correlated_features', {})
        if top_corr_features:
            features = list(top_corr_features.keys())[:10]
            correlations = list(top_corr_features.values())[:10]
            ax4.barh(range(len(features)), correlations, color='lightgreen')
            ax4.set_yticks(range(len(features)))
            ax4.set_yticklabels(features)
            ax4.set_title('Top Features by Correlation')
            ax4.set_xlabel('Correlation with Target')
        
        # Plot 5: Data summary statistics
        ax5 = plt.subplot(3, 3, 5)
        data_summary = result.get('data_summary', {})
        if data_summary:
            metrics = ['Total Samples', 'Total Features', 'Target Mean', 'Target Std']
            values = [
                data_summary.get('total_samples', 0),
                data_summary.get('total_features', 0),
                data_summary.get('target_mean', 0),
                data_summary.get('target_std', 0)
            ]
            ax5.bar(metrics, values, color='orange', alpha=0.7)
            ax5.set_title('Data Summary')
            ax5.tick_params(axis='x', rotation=45)
        
        # Plot 6: Anomaly analysis
        ax6 = plt.subplot(3, 3, 6)
        anomaly_threshold = result.get('anomaly_threshold', 0)
        anomaly_count = (self.y_hist > anomaly_threshold).sum()
        total_count = len(self.y_hist)
        normal_count = total_count - anomaly_count
        
        ax6.pie([normal_count, anomaly_count], labels=['Normal', 'Anomalies'], 
                autopct='%1.1f%%', colors=['lightblue', 'red'])
        ax6.set_title('Anomaly Distribution')
        
        # Plot 7: Feature importance comparison
        ax7 = plt.subplot(3, 3, 7)
        if 'top_correlated_features' in result:
            top_features = list(result['top_correlated_features'].keys())[:8]
            corr_values = list(result['top_correlated_features'].values())[:8]
            ax7.bar(range(len(top_features)), corr_values, color='purple', alpha=0.7)
            ax7.set_xticks(range(len(top_features)))
            ax7.set_xticklabels(top_features, rotation=45, ha='right')
            ax7.set_title('Feature Importance (Correlation)')
            ax7.set_ylabel('Correlation Coefficient')
        
        # Plot 8: Temporal patterns
        ax8 = plt.subplot(3, 3, 8)
        if len(self.y_hist) >= 7:
            rolling_mean = self.y_hist.rolling(window=7).mean()
            ax8.plot(self.y_hist.index, self.y_hist.values, 'b-', alpha=0.6, label='COH')
            ax8.plot(rolling_mean.index, rolling_mean.values, 'r-', linewidth=2, label='7-day rolling mean')
            ax8.set_title('Temporal Patterns')
            ax8.set_xlabel('Date')
            ax8.set_ylabel('COH')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        
        # Plot 9: Summary statistics
        ax9 = plt.subplot(3, 3, 9)
        summary_stats = [
            f"Station: {self.station_code}",
            f"Date: {self.target_date}",
            f"Method: {result['method']}",
            f"Top Root Cause: {result['top_features'][0] if result['top_features'] else 'N/A'}",
            f"Anomaly Threshold: {anomaly_threshold:.2f}",
            f"Today's COH: {self.y_today:.2f}"
        ]
        
        ax9.text(0.1, 0.9, '\n'.join(summary_stats), transform=ax9.transAxes, 
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray'))
        ax9.set_title('Analysis Summary')
        ax9.axis('off')
        
        plt.tight_layout()
        return fig

    def generate_report(self):
        """
        Generate a comprehensive RCA report with data visualization insights.
        
        Returns
        -------
        report : str
            Comprehensive RCA report including visualization insights.
        """
        if not self.fitted:
            self.fit()
        
        # Generate data visualizations
        #viz_report = self.generate_data_visualizations()
        #viz_summary = self.get_visualization_summary()
        
        # Perform attribution analysis
        result = self.analyze_root_causes()
        
        # Generate comprehensive report
        report = f"""
        ROOT CAUSE ANALYSIS REPORT
        ==========================

        Station: {self.station_code}
        Target Date: {self.target_date}
        COH Value: {self.y_today:.2f}
        Analysis Method: {result['method']}

        ATTRIBUTION ANALYSIS RESULTS
        ----------------------------
        Top Root Causes:
        """
                
        for i, (feature, score) in enumerate(zip(result['top_features'], result['top_scores'])):
                    score = score * 100
                    report += f"{i+1}. {feature}: {score:.2f}%\n"
        
        return report


def get_rca_report(station_code: str, target_date: str) -> str:
    """
    Generate Root Cause Attribution (RCA) report for a specific station and date.
    
    Args:
        station_code (str): The station code for analysis.
        target_date (str): The target date for analysis in the format of YYYY-MM-DD.
        
    Returns:
        {
            "status": "success" | "error",
            "report": "The RCA report",
            "error_message": "The error message if the status is error"
        }
    """
    try: 
        rca = StationCOHRCA(station_code, target_date)
        report = rca.generate_report()
        return {
            "status": "success",
            "report": report
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"The error message if the status is error: {e}"
        }

target_station = 'DCM3'
target_date = '2025-07-21'
report = get_rca_report(target_station, target_date)

#report = rca.generate_report()
print(report)