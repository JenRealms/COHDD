"""
Enhanced Data Visualization Module for COH Analysis

This module provides comprehensive data visualization capabilities for the COH business,
supporting multiple chart types including geographic maps, statistical charts, 
time series, and correlation analysis. It integrates with existing station location
data and coordinate functions.
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import warnings
import re
import time
import math
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

# Add the project root to the path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import existing coordinate function
from utils.weather_server import get_city_coordinates

# Configure Plotly for better HTML output
pio.templates.default = "plotly_white"

class DataVisualizer:
    """
    Enhanced data visualization class supporting multiple chart types
    with markdown input parsing and station location integration.
    """
    
    def __init__(self):
        """Initialize the visualizer with station data and geocoding setup."""
        self.station_data = None
        self.geocoder = Nominatim(user_agent="coh_visualization")
        self.coordinate_cache = {}
        self._load_station_data()
    
    def _load_station_data(self):
        """Load and filter station location data."""
        try:
            station_file = os.path.join(project_root, "data", "Station_Locations.csv")
            df = pd.read_csv(station_file)
            
            # Filter out stations with empty city or state
            df = df.dropna(subset=['City', 'State'])
            df = df[df['City'].str.strip() != '']
            df = df[df['State'].str.strip() != '']
            
            self.station_data = df
            print(f"Loaded {len(self.station_data)} valid stations")
            
        except Exception as e:
            print(f"Error loading station data: {e}")
            self.station_data = pd.DataFrame()
    
    def get_station_coordinates(self, city: str, state: str) -> Tuple[float, float]:
        """
        Get coordinates for a city/state combination with caching.
        
        Args:
            city: City name (str)
            state: State name (str)
            
        Returns:
            Tuple of (latitude, longitude)
        """
        cache_key = f"{city}_{state}".lower()
        
        if cache_key in self.coordinate_cache:
            return self.coordinate_cache[cache_key]
        
        try:
            # Try existing function first
            coords = get_city_coordinates(city, state)
            if coords and len(coords) == 2:
                self.coordinate_cache[cache_key] = coords
                return coords
        except:
            pass
        
        # Fallback to free geocoding service
        try:
            location = self.geocoder.geocode(f"{city}, {state}, USA")
            if location:
                coords = (location.latitude, location.longitude)
                self.coordinate_cache[cache_key] = coords
                time.sleep(1)  # Rate limiting for free service
                return coords
        except (GeocoderTimedOut, GeocoderUnavailable) as e:
            print(f"Geocoding error for {city}, {state}: {e}")
        
        # Return None if geocoding fails
        return None
    
    def parse_markdown_data(self, markdown_text: str) -> pd.DataFrame:
        """
        Parse markdown data into a pandas DataFrame.
        
        Args:
            markdown_text: Markdown formatted data
            
        Returns:
            Parsed DataFrame
        """
        try:
            # Split into lines and clean up
            lines = [line.strip() for line in markdown_text.split('\n') if line.strip()]
            
            # Find the separator line (contains dashes and colons)
            separator_idx = None
            for i, line in enumerate(lines):
                if '|' in line and ('-' in line or ':' in line):
                    separator_idx = i
                    break
            
            if separator_idx is None:
                print("No table separator line found")
                return pd.DataFrame()
            
            # Get headers from the line before separator
            if separator_idx > 0:
                header_line = lines[separator_idx - 1]
                headers = [h.strip() for h in header_line.split('|') if h.strip()]
            else:
                print("No header line found")
                return pd.DataFrame()
            
            # Get data rows after separator
            data_rows = []
            for line in lines[separator_idx + 1:]:
                if '|' in line:
                    # Split by | and clean each cell
                    cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                    if len(cells) == len(headers):
                        data_rows.append(cells)
            
            if not data_rows:
                print("No data rows found")
                return pd.DataFrame()
            
            # Create DataFrame
            df = pd.DataFrame(data_rows, columns=headers)
            print(f"Successfully parsed table with {len(df)} rows and {len(df.columns)} columns")
            
            return self._convert_data_types(df)
            
        except Exception as e:
            print(f"Error parsing markdown data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert data types in the DataFrame, especially numeric columns."""
        try:
            # Make a copy to avoid modifying the original
            df_converted = df.copy()
            
            for col in df_converted.columns:
                # Skip if column is empty or all NaN
                if df_converted[col].isna().all():
                    continue
                
                # Skip station code columns - they should remain as strings
                if col.lower() in ['station_code', 'station_code', 'station']:
                    continue
                
                # Try to convert to numeric only for non-station code columns
                try:
                    # Remove any non-numeric characters except decimal points and minus signs
                    cleaned_series = df_converted[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                    # Convert to numeric, coercing errors to NaN
                    numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
                    
                    # Only convert if we have valid numeric data
                    if not numeric_series.isna().all():
                        df_converted[col] = numeric_series
                        print(f"Converted column '{col}' to numeric")
                except:
                    # If conversion fails, keep as string
                    pass
            
            return df_converted
            
        except Exception as e:
            print(f"Error converting data types: {e}")
            return df
    
    def _extract_structured_data(self, text: str) -> pd.DataFrame:
        """Extract structured data from text when no clear table format is found."""
        lines = text.strip().split('\n')
        data = []
        
        for line in lines:
            if ':' in line and not line.startswith('#'):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    data.append([key, value])
        
        if data:
            return pd.DataFrame(data, columns=['Key', 'Value'])
        
        return pd.DataFrame()
    
    def detect_data_type(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Detect data types in the DataFrame to help with chart selection.
        
        Args:
            df: Input DataFrame (pandas.DataFrame)
            
        Returns:
            Dictionary with data type information
        """
        data_info = {
            'has_numeric': False,
            'has_categorical': False,
            'has_temporal': False,
            'has_geographic': False,
            'columns': list(df.columns),
            'shape': df.shape
        }
        
        for col in df.columns:
            # Check for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                data_info['has_numeric'] = True
            elif df[col].dtype == 'object':
                # Check if it's actually numeric
                try:
                    pd.to_numeric(df[col], errors='raise')
                    data_info['has_numeric'] = True
                except:
                    # Check for temporal data
                    try:
                        pd.to_datetime(df[col], format='mixed', errors='raise')
                        data_info['has_temporal'] = True
                    except:
                        # Check for geographic data
                        if any(keyword in col.lower() for keyword in ['city', 'state', 'station', 'location']):
                            data_info['has_geographic'] = True
                        else:
                            data_info['has_categorical'] = True
        
        return data_info
    
    def suggest_chart_type(self, df: pd.DataFrame, data_info: Dict[str, str]) -> str:
        """
        Suggest the most appropriate chart type based on data structure.
        
        Args:
            df: Input DataFrame
            data_info: Data type information
            
        Returns:
            Suggested chart type
        """
        if data_info['has_geographic'] and data_info['has_numeric']:
            return 'geographic_map'
        elif data_info['has_temporal'] and data_info['has_numeric']:
            if len([col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]) >= 2:
                return 'double_y_time_series'
            else:
                return 'time_series'
        elif data_info['has_categorical'] and data_info['has_numeric']:
            if len(df) <= 20:
                return 'bar_chart'
            else:
                return 'histogram'
        elif data_info['has_numeric']:
            if len([col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]) >= 2:
                return 'scatter_plot'
            else:
                return 'distribution_plot'
        else:
            return 'table_view'
    
    def create_geographic_map(self, df: pd.DataFrame, 
                             value_column: str, 
                             title: str = "Station Map") -> go.Figure:
        """
        Create a geographic map visualization.
        
        Args:
            df: DataFrame with geographic data
            value_column: Column containing values to visualize
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        # Check if we already have coordinates
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            # Use existing coordinates
            coordinates = []
            valid_stations = []
            
            for _, row in df.iterrows():
                if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
                    coordinates.append((row['Latitude'], row['Longitude']))
                    valid_stations.append(row)
        else:
            # Check if we have City/State data directly
            if 'City' in df.columns and 'State' in df.columns:
                # Use City/State data directly
                coordinates = []
                valid_stations = []
                
                for _, row in df.iterrows():
                    if pd.notna(row['City']) and pd.notna(row['State']):
                        coords = self.get_station_coordinates(row['City'], row['State'])
                        if coords:
                            coordinates.append(coords)
                            valid_stations.append(row)
            else:
                # Fallback to original method - merge with station location data
                merged_df = df.merge(self.station_data, left_on='Station_Code', right_on='Station_Code', how='inner')
                
                # Get coordinates for each station
                coordinates = []
                valid_stations = []
                
                for _, row in merged_df.iterrows():
                    coords = self.get_station_coordinates(row['City'], row['State'])
                    if coords:
                        coordinates.append(coords)
                        valid_stations.append(row)
        
        if not coordinates:
            raise ValueError("No valid coordinates found for stations")
        
        # Create the map
        lats = [coord[0] for coord in coordinates]
        lons = [coord[1] for coord in coordinates]
        
        # Extract values and station names from valid_stations
        values = []
        station_names = []
        
        for station in valid_stations:
            if value_column in station and pd.notna(station[value_column]):
                values.append(station[value_column])
                # Get station name from available columns
                if 'City' in station and 'State' in station:
                    station_names.append(f"{station['City']}, {station['State']}")
                elif 'station_code' in station:
                    station_names.append(f"Station {station['station_code']}")
                else:
                    station_names.append(f"Station {len(station_names) + 1}")
            else:
                # Skip stations without valid values
                continue
        
        # Remove corresponding coordinates for stations without values
        if len(values) != len(coordinates):
            # Rebuild coordinates list to match values
            valid_coordinates = []
            for i, station in enumerate(valid_stations):
                if value_column in station and pd.notna(station[value_column]):
                    valid_coordinates.append(coordinates[i])
            coordinates = valid_coordinates
            lats = [coord[0] for coord in coordinates]
            lons = [coord[1] for coord in coordinates]
        
        if not values:
            raise ValueError(f"No valid values found in column '{value_column}'")
        
        fig = go.Figure()
        
        # Add scatter plot for stations
        fig.add_trace(go.Scattergeo(
            lon=lons,
            lat=lats,
            mode='markers',
            marker=dict(
                size=8,
                color=values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=value_column)
            ),
            text=station_names,
            hovertemplate='<b>%{text}</b><br>' +
                         f'{value_column}: %{{marker.color}}<br>' +
                         'Lat: %{lat:.2f}<br>' +
                         'Lon: %{lon:.2f}<extra></extra>'
        ))
        
        # Update layout for US map
        fig.update_geos(
            scope='usa',
            projection_type='albers usa',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            coastlinecolor='rgb(204, 204, 204)',
            showocean=True,
            oceancolor='rgb(230, 230, 250)'
        )
        
        fig.update_layout(
            title=title,
            height=600,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        return fig
    
    def create_time_series(self, markdown_data: str,
                          value_columns: list,
                          title: str = "Time Series",
                          x_axis_title: Optional[str] = None,
                          y_axis_title: Optional[str] = None,
                          show_legend: bool = True,
                          line_mode: str = "lines+markers",
                          line_width: int = 2,
                          marker_size: int = 6,
                          height: int = 600,
                          width: Optional[int] = None,
                          color_scheme: str = "plotly",
                          hover_mode: str = "x unified",
                          grid_style: str = "light") -> Dict[str, str]:
        """
        Create a time series chart from markdown data.  
        
        Args:
            markdown_data: Markdown formatted data. The first column is datetime (YYYY-MM-DD format) and the rest of the columns are the value columns.
            value_columns: List of column names (strings) containing numeric values to plot
            title: Chart title
            x_axis_title: Custom title for x-axis (defaults to "Date" if None)
            y_axis_title: Custom title for y-axis (defaults to "Value" if None)
            show_legend: (Optional) Whether to show the legend (boolean)
            line_mode: (Optional) Plot mode - "lines", "markers", or "lines+markers"
            line_width: (Optional) Width of the lines
            marker_size: (Optional) Size of the markers
            height: (Optional) Chart height in pixels
            width: (Optional) Chart width in pixels (None for auto)
            color_scheme: (Optional) Color scheme for the lines ("plotly", "viridis", "plasma", etc.)
            hover_mode: (Optional) Hover mode - "x unified", "y unified", "closest", or "x"
            grid_style: (Optional) Grid style - "light", "dark", "white", or "none"
            
            If the args are not labeled Optional, they are required.
        Returns:
            { "status": "success" | "error",  
              "html_content": HTML content as string,
              "full_path": file path where the html chart is saved (string),
              "error_message": "The error message if the status is error" }
        """
        # Parse markdown data
        df = self.parse_markdown_data(markdown_data)
        
        if df.empty:
            raise ValueError("No data could be parsed from markdown input")
        
        # Validate that we have at least one value column
        if not value_columns or not isinstance(value_columns, list):
            raise ValueError("value_columns must be a non-empty list of column names")
        
        # Check if all value columns exist in the data
        missing_columns = [col for col in value_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Value columns not found in data: {missing_columns}")
        
        # Get the first column as the datetime column
        datetime_column = df.columns[0]
        
        # Convert datetime column to proper format
        df_copy = df.copy()
        df_copy[datetime_column] = pd.to_datetime(df_copy[datetime_column], format='%Y-%m-%d', errors='coerce')
        
        # Remove rows with invalid dates
        df_copy = df_copy.dropna(subset=[datetime_column])
        
        if df_copy.empty:
            raise ValueError("No valid dates found in the first column")
        
        # Sort by datetime
        df_copy = df_copy.sort_values(datetime_column)
        
        # Create the figure
        fig = go.Figure()
        
        # Generate colors for multiple lines
        if color_scheme == "plotly":
            colors = px.colors.qualitative.Plotly
        elif color_scheme == "viridis":
            colors = px.colors.sequential.Viridis
        elif color_scheme == "plasma":
            colors = px.colors.sequential.Plasma
        else:
            colors = px.colors.qualitative.Set1
        
        # Add traces for each value column
        for i, column in enumerate(value_columns):
            # Get color for this line (cycle through colors if more columns than colors)
            color = colors[i % len(colors)]
            
            # Convert to numeric, handling any non-numeric values
            numeric_values = pd.to_numeric(df_copy[column], errors='coerce')
            
            # Remove rows with NaN values for this column
            valid_data = df_copy.dropna(subset=[column])
            
            if not valid_data.empty:
                fig.add_trace(go.Scatter(
                    x=valid_data[datetime_column],
                    y=valid_data[column],
                    mode=line_mode,
                    name=column,
                    line=dict(width=line_width, color=color),
                    marker=dict(size=marker_size, color=color),
                    hovertemplate=f'<b>{column}</b><br>' +
                                'Date: %{x|%Y-%m-%d}<br>' +
                                'Value: %{y:.2f}<extra></extra>'
                ))
        
        # Set default axis titles if not provided
        if x_axis_title is None:
            x_axis_title = "Date"
        if y_axis_title is None:
            y_axis_title = "Value"
        
        # Update layout with enhanced styling
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=18, color='#2c3e50', weight='bold')
            ),
            xaxis_title=x_axis_title,
            yaxis_title=y_axis_title,
            hovermode=hover_mode,
            height=height,
            width=width,
            showlegend=show_legend,
            legend=dict(
                x=1.02,
                y=1,
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='#bdc3c7',
                borderwidth=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Arial, sans-serif', size=12, color='#2c3e50'),
            margin=dict(l=60, r=80, t=60, b=60)
        )
        
        # Update axes styling
        fig.update_xaxes(
            gridcolor='#ecf0f1',
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor='#bdc3c7',
            linewidth=1
        )
        
        fig.update_yaxes(
            gridcolor='#ecf0f1',
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor='#bdc3c7',
            linewidth=1
        )

        # Export to HTML
        html_content, full_path = self.export_to_html(fig, f"{title}.html")
        
        return {
            "status": "success",
            "html_content": html_content,
            "full_path": full_path
        }
    
    def create_double_y_time_series(self, markdown_data: str,
                                   y1_column: str,
                                   y2_column: str,
                                   title: str = "Double Y-Axis Time Series",
                                   x_axis_title: Optional[str] = None,
                                   y1_axis_title: Optional[str] = None,
                                   y2_axis_title: Optional[str] = None,
                                   show_legend: bool = True,
                                   line_mode: str = "lines+markers",
                                   line_width: int = 2,
                                   marker_size: int = 6,
                                   height: int = 600,
                                   width: Optional[int] = None,
                                   color_scheme: str = "custom",
                                   hover_mode: str = "x unified",
                                   grid_style: str = "light") -> Dict[str, str]:
        """
        Create a time series chart with two y-axes from markdown data.
        This chart plots exactly ONE metric on each y-axis for comparison.
        You will use this function to plot two time series with different units or scales.
        For example, you can plot COH (0-72) and utilization (%) on the same chart.
        
        Args:
            markdown_data: Markdown formatted data. The first column is datetime (YYYY-MM-DD format) and the rest 2 columns are the value columns.
            y1_column: Column name for the SINGLE metric to plot on the left y-axis
            y2_column: Column name for the SINGLE metric to plot on the right y-axis
            title: Chart title
            x_axis_title: (Optional) Custom title for x-axis (defaults to "Date" if None)
            y1_axis_title: (Optional) Custom title for left y-axis (defaults to y1_column if None)
            y2_axis_title: (Optional) Custom title for right y-axis (defaults to y2_column if None)
            show_legend: (Optional) Whether to show the legend (boolean)
            line_mode: (Optional) Plot mode - "lines", "markers", or "lines+markers" (string)
            line_width: (Optional) Width of the lines (integer)
            marker_size: (Optional) Size of the markers (integer)
            height: (Optional) Chart height in pixels (integer)
            width: (Optional) Chart width in pixels (None for auto)
            color_scheme: (Optional) Color scheme - "custom" (blue/red), "plotly", "viridis", or "plasma" (string)
            hover_mode: (Optional) Hover mode - "x unified", "y unified", "closest", or "x" (string)
            grid_style: (Optional) Grid style - "light", "dark", "white", or "none" (string)
            
            If the args are not labeled Optional, they are required.
        Returns:
            { "status": "success" | "error",  
              "html_content": HTML content as string,
              "full_path": file path where the html chart is saved (string),
              "error_message": "The error message if the status is error" }
        """
        # Parse markdown data
        df = self.parse_markdown_data(markdown_data)
        
        if df.empty:
            return {
                "status": "error",
                "error_message": "No data could be parsed from markdown input"
            }
        
        # Validate that we have the required columns
        if y1_column not in df.columns:
            return {
                "status": "error",
                "error_message": f"Y1 column '{y1_column}' not found in data"
            }
        
        if y2_column not in df.columns:
            return {
                "status": "error",
                "error_message": f"Y2 column '{y2_column}' not found in data"
            }
        
        # Get the first column as the datetime column
        datetime_column = df.columns[0]
        
        # Convert datetime column to proper format
        df_copy = df.copy()
        df_copy[datetime_column] = pd.to_datetime(df_copy[datetime_column], format='%Y-%m-%d', errors='coerce')
        
        # Remove rows with invalid dates
        df_copy = df_copy.dropna(subset=[datetime_column])
        
        if df_copy.empty:
            return {
                "status": "error",
                "error_message": "No valid dates found in the first column"
            }
        
        # Sort by datetime
        df_copy = df_copy.sort_values(datetime_column)
        
        # Create the figure
        try:
            fig = go.Figure()
        except Exception as e:
            return {
                "status": "error",
                "error_message": f"Error creating figure: {str(e)}"
            }
        
        # Set default axis titles if not provided
        if x_axis_title is None:
            x_axis_title = "Date"
        if y1_axis_title is None:
            y1_axis_title = y1_column
        if y2_axis_title is None:
            y2_axis_title = y2_column
        
        # Determine colors based on color scheme
        if color_scheme == "custom":
            y1_color = 'blue'
            y2_color = 'red'
        elif color_scheme == "plotly":
            colors = px.colors.qualitative.Plotly
            y1_color = colors[0]
            y2_color = colors[1]
        elif color_scheme == "viridis":
            colors = px.colors.sequential.Viridis
            y1_color = colors[0]
            y2_color = colors[-1]
        elif color_scheme == "plasma":
            colors = px.colors.sequential.Plasma
            y1_color = colors[0]
            y2_color = colors[-1]
        else:
            colors = px.colors.qualitative.Set1
            y1_color = colors[0]
            y2_color = colors[1]
        
        # Convert to numeric, handling any non-numeric values
        y1_values = pd.to_numeric(df_copy[y1_column], errors='coerce')
        y2_values = pd.to_numeric(df_copy[y2_column], errors='coerce')
        
        # Remove rows with NaN values for either column
        valid_data = df_copy.dropna(subset=[y1_column, y2_column])
        
        if valid_data.empty:
            return {
                "status": "error",
                "error_message": f"No valid numeric data found in columns '{y1_column}' and '{y2_column}'"
            }
        
        # First y-axis (left)
        fig.add_trace(go.Scatter(
            x=valid_data[datetime_column],
            y=valid_data[y1_column],
            mode=line_mode,
            name=y1_column,
            line=dict(width=line_width, color=y1_color),
            marker=dict(size=marker_size, color=y1_color),
            yaxis='y',
            hovertemplate=f'<b>{y1_column}</b><br>' +
                          'Date: %{x|%Y-%m-%d}<br>' +
                          'Value: %{y:.2f}<extra></extra>'
        ))
        
        # Second y-axis (right)
        fig.add_trace(go.Scatter(
            x=valid_data[datetime_column],
            y=valid_data[y2_column],
            mode=line_mode,
            name=y2_column,
            line=dict(width=line_width, color=y2_color),
            marker=dict(size=marker_size, color=y2_color),
            yaxis='y2',
            hovertemplate=f'<b>{y2_column}</b><br>' +
                          'Date: %{x|%Y-%m-%d}<br>' +
                          'Value: %{y:.2f}<extra></extra>'
        ))
        
        # Update layout with enhanced styling
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=18, color='#2c3e50', weight='bold')
            ),
            xaxis_title=x_axis_title,
            yaxis=dict(
                title=y1_axis_title,
                side='left',
                gridcolor='#ecf0f1',
                gridwidth=1,
                zeroline=False,
                showline=True,
                linecolor='#bdc3c7',
                linewidth=1
            ),
            yaxis2=dict(
                title=y2_axis_title,
                side='right',
                overlaying='y',
                gridcolor='#ecf0f1',
                gridwidth=1,
                zeroline=False,
                showline=True,
                linecolor='#bdc3c7',
                linewidth=1
            ),
            hovermode=hover_mode,
            height=height,
            width=width,
            showlegend=show_legend,
            legend=dict(
                x=1.02,
                y=1,
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='#bdc3c7',
                borderwidth=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Arial, sans-serif', size=12, color='#2c3e50'),
            margin=dict(l=60, r=80, t=60, b=60)
        )
        
        # Update x-axis styling
        fig.update_xaxes(
            gridcolor='#ecf0f1',
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor='#bdc3c7',
            linewidth=1
        )
        
        # Export to HTML
        html_content, full_path = self.export_to_html(fig, f"{title}.html")
        
        return {
            "status": "success",
            "html_content": html_content,
            "full_path": full_path
        }
    
    def create_correlation_plot(self, df: pd.DataFrame,
                               title: str = "Correlation Matrix") -> go.Figure:
        """
        Create a correlation heatmap.
        
        Args:
            df: DataFrame with numeric data
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            raise ValueError("No numeric columns found for correlation analysis")
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title=title,
            height=500,
            xaxis_title="Variables",
            yaxis_title="Variables"
        )
        
        return fig
    
    def create_distribution_plot(self, markdown_data: str,
                                column: str,
                                plot_type: str = "histogram",
                                title: Optional[str] = None,
                                x_axis_title: Optional[str] = None,
                                y_axis_title: Optional[str] = None,
                                show_legend: bool = True,
                                height: int = 600,
                                width: Optional[int] = None,
                                color_scheme: str = "plotly",
                                grid_style: str = "light",
                                bins: Optional[int] = None,
                                opacity: float = 0.7) -> Dict[str, str]:
        """
        Create distribution plots (histogram, box plot, violin plot) from markdown data.
        This function creates various types of distribution visualizations for a single metric.
        
        Args:
            markdown_data: Markdown formatted data. The first column is typically the metric to plot
            column: Column name (string) containing the numeric values to visualize
            plot_type: (Optional) Type of distribution plot - "histogram" (default), "box", or "violin" (string)
            title: (Optional) Chart title (string, defaults to auto-generated title if None)
            x_axis_title: (Optional) Custom title for x-axis (string, defaults to column name if None)
            y_axis_title: (Optional) Custom title for y-axis (string, defaults to "Count" or "Value" if None)
            show_legend: (Optional) Whether to show the legend (boolean)
            height: (Optional) Chart height in pixels (integer)
            width: (Optional) Chart width in pixels (None for auto)
            color_scheme: (Optional) Color scheme for the plot (string)
            grid_style: (Optional) Grid style - "light", "dark", "white", or "none" (string)
            bins: (Optional) Number of bins for histogram (integer, only applies to histogram)
            opacity: (Optional) Opacity of the plot elements (float, 0.0 to 1.0)
            
            If the args are not labeled Optional, they are required.
        Returns:
            { "status": "success" | "error",  
              "html_content": HTML content as string,
              "full_path": file path where the html chart is saved (string),
              "error_message": "The error message if the status is error" }
        """
        # Parse markdown data
        df = self.parse_markdown_data(markdown_data)
        
        if df.empty:
            return {
                "status": "error",
                "error_message": "No data could be parsed from markdown input"
            }
        
        # Validate that the column exists
        if column not in df.columns:
            return {
                "status": "error",
                "error_message": f"Column '{column}' not found in data. Available columns: {list(df.columns)}"
            }
        
        # Convert to numeric, handling any non-numeric values
        numeric_values = pd.to_numeric(df[column], errors='coerce')
        
        # Remove rows with NaN values
        valid_data = df.dropna(subset=[column])
        
        if valid_data.empty:
            return {
                "status": "error",
                "error_message": f"No valid numeric data found in column '{column}'"
            }
        
        # Set default title if not provided
        if title is None:
            title = f"{plot_type.title()} of {column}"
        
        # Set default axis titles if not provided
        if x_axis_title is None:
            x_axis_title = column
        if y_axis_title is None:
            if plot_type == "histogram":
                y_axis_title = "Count"
            else:
                y_axis_title = "Value"
        
        # Create the figure
        try:
            fig = go.Figure()
        except Exception as e:
            return {
                "status": "error",
                "error_message": f"Error creating figure: {str(e)}"
            }
        
        # Determine colors based on color scheme
        if color_scheme == "plotly":
            colors = px.colors.qualitative.Plotly
            plot_color = colors[0]
        elif color_scheme == "viridis":
            colors = px.colors.sequential.Viridis
            plot_color = colors[len(colors)//2]
        elif color_scheme == "plasma":
            colors = px.colors.sequential.Plasma
            plot_color = colors[len(colors)//2]
        else:
            colors = px.colors.qualitative.Set1
            plot_color = colors[0]
        
        # Create the appropriate plot type
        if plot_type == "histogram":
            # Use Plotly Express for histogram with custom styling
            fig = px.histogram(
                valid_data, 
                x=column, 
                title=title,
                height=height,
                width=width,
                nbins=bins,
                opacity=opacity,
                color_discrete_sequence=[plot_color]
            )
            
            # Update layout for better styling
            fig.update_layout(
                title=dict(
                    text=title,
                    x=0.5,
                    xanchor='center',
                    font=dict(size=18, color='#2c3e50', weight='bold')
                ),
                xaxis_title=x_axis_title,
                yaxis_title=y_axis_title,
                showlegend=show_legend,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Arial, sans-serif', size=12, color='#2c3e50'),
                margin=dict(l=60, r=60, t=60, b=60)
            )
            
            # Update axes styling
            fig.update_xaxes(
                gridcolor='#ecf0f1',
                gridwidth=1,
                zeroline=False,
                showline=True,
                linecolor='#bdc3c7',
                linewidth=1
            )
            
            fig.update_yaxes(
                gridcolor='#ecf0f1',
                gridwidth=1,
                zeroline=False,
                showline=True,
                linecolor='#bdc3c7',
                linewidth=1
            )
            
        elif plot_type == "box":
            # Use Plotly Express for box plot with custom styling
            fig = px.box(
                valid_data, 
                y=column, 
                title=title,
                height=height,
                width=width,
                color_discrete_sequence=[plot_color]
            )
            
            # Update layout for better styling
            fig.update_layout(
                title=dict(
                    text=title,
                    x=0.5,
                    xanchor='center',
                    font=dict(size=18, color='#2c3e50', weight='bold')
                ),
                xaxis_title=x_axis_title,
                yaxis_title=y_axis_title,
                showlegend=show_legend,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Arial, sans-serif', size=12, color='#2c3e50'),
                margin=dict(l=60, r=60, t=60, b=60)
            )
            
            # Update axes styling
            fig.update_xaxes(
                gridcolor='#ecf0f1',
                gridwidth=1,
                zeroline=False,
                showline=True,
                linecolor='#bdc3c7',
                linewidth=1
            )
            
            fig.update_yaxes(
                gridcolor='#ecf0f1',
                gridwidth=1,
                zeroline=False,
                showline=True,
                linecolor='#bdc3c7',
                linewidth=1
            )
            
        elif plot_type == "violin":
            # Use Plotly Express for violin plot with custom styling
            fig = px.violin(
                valid_data, 
                y=column, 
                title=title,
                height=height,
                width=width,
                color_discrete_sequence=[plot_color]
            )
            
            # Update layout for better styling
            fig.update_layout(
                title=dict(
                    text=title,
                    x=0.5,
                    xanchor='center',
                    font=dict(size=18, color='#2c3e50', weight='bold')
                ),
                xaxis_title=x_axis_title,
                yaxis_title=y_axis_title,
                showlegend=show_legend,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Arial, sans-serif', size=12, color='#2c3e50'),
                margin=dict(l=60, r=60, t=60, b=60)
            )
            
            # Update axes styling
            fig.update_xaxes(
                gridcolor='#ecf0f1',
                gridwidth=1,
                zeroline=False,
                showline=True,
                linecolor='#bdc3c7',
                linewidth=1
            )
            
            fig.update_yaxes(
                gridcolor='#ecf0f1',
                gridwidth=1,
                zeroline=False,
                showline=True,
                linecolor='#bdc3c7',
                linewidth=1
            )
            
        else:
            return {
                "status": "error",
                "error_message": f"Unsupported plot type: {plot_type}. Supported types: histogram, box, violin"
            }
        
        # Export to HTML
        html_content, full_path = self.export_to_html(fig, f"{title}.html")
        
        return {
            "status": "success",
            "html_content": html_content,
            "full_path": full_path
        }
    
    def create_bar_chart(self, markdown_data: str,
                         x_column: str,
                         y_column: str,
                         title: str = "Bar Chart",
                         x_axis_title: Optional[str] = None,
                         y_axis_title: Optional[str] = None,
                         show_legend: bool = True,
                         height: int = 600,
                         width: Optional[int] = None,
                         color_scheme: str = "plotly",
                         orientation: str = "v",
                         opacity: float = 0.8,
                         text_auto: str = "auto",
                         show_text: bool = False) -> Dict[str, str]:
        """
        Create a bar chart from markdown data.
        This function creates vertical or horizontal bar charts for comparing values across categories.
        
        Args:
            markdown_data: Markdown formatted data with categorical and numeric columns
            x_column: Column name (string) for x-axis (categories)
            y_column: Column name (string) for y-axis (numeric values)
            title: Chart title (string)
            x_axis_title: Custom title for x-axis (string, defaults to x_column if None)
            y_axis_title: Custom title for y-axis (string, defaults to y_column if None)
            show_legend: (Optional)Whether to show the legend (boolean)
            height: (Optional) Chart height in pixels (integer)
            width: (Optional) Chart width in pixels (None for auto)
            color_scheme: (Optional) Color scheme for the bars (string)
            orientation: (Optional) Bar orientation - "v" for vertical, "h" for horizontal (string)
                
            If the args are not labeled Optional, they are required.
        Returns:
            { "status": "success" | "error",  
              "html_content": HTML content as string,
              "full_path": file path where the html chart is saved (string),
              "error_message": "The error message if the status is error" }
        """
        # Parse markdown data
        df = self.parse_markdown_data(markdown_data)
        
        if df.empty:
            return {
                "status": "error",
                "error_message": "No data could be parsed from markdown input"
            }
        
        # Validate that the required columns exist
        if x_column not in df.columns:
            return {
                "status": "error",
                "error_message": f"X column '{x_column}' not found in data. Available columns: {list(df.columns)}"
            }
        
        if y_column not in df.columns:
            return {
                "status": "error",
                "error_message": f"Y column '{y_column}' not found in data. Available columns: {list(df.columns)}"
            }
        
        # Convert y_column to numeric, handling any non-numeric values
        numeric_values = pd.to_numeric(df[y_column], errors='coerce')
        
        # Remove rows with NaN values in y_column
        valid_data = df.dropna(subset=[y_column])
        
        if valid_data.empty:
            return {
                "status": "error",
                "error_message": f"No valid numeric data found in column '{y_column}'"
            }
        
        # Set default axis titles if not provided
        if x_axis_title is None:
            x_axis_title = x_column
        if y_axis_title is None:
            y_axis_title = y_column
        
        # Create the figure
        try:
            fig = go.Figure()
        except Exception as e:
            return {
                "status": "error",
                "error_message": f"Error creating figure: {str(e)}"
            }
        
        # Determine colors based on color scheme
        if color_scheme == "plotly":
            colors = px.colors.qualitative.Plotly
            bar_color = colors[0]
        elif color_scheme == "viridis":
            colors = px.colors.sequential.Viridis
            bar_color = colors[len(colors)//2]
        elif color_scheme == "plasma":
            colors = px.colors.sequential.Plasma
            bar_color = colors[len(colors)//2]
        elif color_scheme == "custom":
            bar_color = "#3498db"  # Professional blue
        else:
            colors = px.colors.qualitative.Set1
            bar_color = colors[0]
        
        # Create the bar chart
        if orientation == "h":
            # Horizontal bar chart
            fig = px.bar(
                valid_data,
                x=y_column,
                y=x_column,
                title=title,
                height=height,
                width=width,
                color_discrete_sequence=[bar_color],
                opacity=opacity,
                text=y_column if show_text else None,
                text_auto=text_auto
            )
            
            # Update layout for horizontal bars
            fig.update_layout(
                title=dict(
                    text=title,
                    x=0.5,
                    xanchor='center',
                    font=dict(size=18, color='#2c3e50', weight='bold')
                ),
                xaxis_title=y_axis_title,
                yaxis_title=x_axis_title,
                showlegend=show_legend,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Arial, sans-serif', size=12, color='#2c3e50'),
                margin=dict(l=60, r=60, t=60, b=60)
            )
            
            # Update axes styling for horizontal bars
            fig.update_xaxes(
                gridcolor='#ecf0f1',
                gridwidth=1,
                zeroline=False,
                showline=True,
                linecolor='#bdc3c7',
                linewidth=1
            )
            
            fig.update_yaxes(
                gridcolor='#ecf0f1',
                gridwidth=1,
                zeroline=False,
                showline=True,
                linecolor='#bdc3c7',
                linewidth=1
            )
            
        else:
            # Vertical bar chart (default)
            fig = px.bar(
                valid_data,
                x=x_column,
                y=y_column,
                title=title,
                height=height,
                width=width,
                color_discrete_sequence=[bar_color],
                opacity=opacity,
                text=y_column if show_text else None,
                text_auto=text_auto
            )
            
            # Update layout for vertical bars
            fig.update_layout(
                title=dict(
                    text=title,
                    x=0.5,
                    xanchor='center',
                    font=dict(size=18, color='#2c3e50', weight='bold')
                ),
                xaxis_title=x_axis_title,
                yaxis_title=y_axis_title,
                showlegend=show_legend,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Arial, sans-serif', size=12, color='#2c3e50'),
                margin=dict(l=60, r=60, t=60, b=60)
            )
            
            # Update axes styling for vertical bars
            fig.update_xaxes(
                gridcolor='#ecf0f1',
                gridwidth=1,
                zeroline=False,
                showline=True,
                linecolor='#bdc3c7',
                linewidth=1
            )
            
            fig.update_yaxes(
                gridcolor='#ecf0f1',
                gridwidth=1,
                zeroline=False,
                showline=True,
                linecolor='#bdc3c7',
                linewidth=1
            )
        
        # Export to HTML
        html_content, full_path = self.export_to_html(fig, f"{title}.html")
        
        return {
            "status": "success",
            "html_content": html_content,
            "full_path": full_path
        }
    
    def create_scatter_plot(self, markdown_data: str,
                           x_column: str,
                           y_column: str,
                           size_column: Optional[str] = None,
                           title: str = "Scatter Plot",
                           x_axis_title: Optional[str] = None,
                           y_axis_title: Optional[str] = None,
                           show_legend: bool = True,
                           height: int = 600,
                           width: Optional[int] = None,
                           color_scheme: str = "plotly",
                           marker_size_range: List[int] = [8, 20],
                           opacity: float = 0.7,
                           show_grid: bool = True) -> Dict[str, str]:
        """
        Create a scatter plot from markdown data.
        This function creates scatter plots with up to 3 columns:
        - First column (x_column): X-axis location
        - Second column (y_column): Y-axis location  
        - Third column (size_column): Size of the dots (optional)
        
        Args:
            markdown_data: Markdown formatted data you want to present.  All the columns are numeric.
            x_column: Column name (string) for x-axis coordinates
            y_column: Column name (string) for y-axis coordinates
            size_column: Column name (string) for dot sizes (optional, defaults to None)
            title: Chart title (string)
            x_axis_title: Custom title for x-axis (string, defaults to x_column if None)
            y_axis_title: Custom title for y-axis (string, defaults to y_column if None)
            show_legend: (Optional) Whether to show the legend (boolean)
            height: (Optional) Chart height in pixels (integer)
            width: (Optional) Chart width in pixels (None for auto)
            color_scheme: (Optional) Color scheme for the dots (string)
            marker_size_range: (Optional) Tuple of (min_size, max_size) for dot sizing (tuple)
            opacity: (Optional) Opacity of the dots (float, 0.0 to 1.0)
            show_grid: (Optional) Whether to show grid lines (boolean)
            
            If the args are not labeled Optional, they are required.
        Returns:
            { "status": "success" | "error",  
              "html_content": HTML content as string,
              "full_path": file path where the html chart is saved (string),
              "error_message": "The error message if the status is error" }
        """
        # Parse markdown data
        df = self.parse_markdown_data(markdown_data)
        
        if df.empty:
            return {
                "status": "error",
                "error_message": "No data could be parsed from markdown input"
            }
        
        # Validate that the required columns exist
        if x_column not in df.columns:
            return {
                "status": "error",
                "error_message": f"X column '{x_column}' not found in data. Available columns: {list(df.columns)}"
            }
        
        if y_column not in df.columns:
            return {
                "status": "error",
                "error_message": f"Y column '{y_column}' not found in data. Available columns: {list(df.columns)}"
            }
        
        # Validate size column if provided
        if size_column and size_column not in df.columns:
            return {
                "status": "error",
                "error_message": f"Size column '{size_column}' not found in data. Available columns: {list(df.columns)}"
            }
        
        # Convert columns to numeric, handling any non-numeric values
        x_values = pd.to_numeric(df[x_column], errors='coerce')
        y_values = pd.to_numeric(df[y_column], errors='coerce')
        
        # Remove rows with NaN values in x or y columns
        valid_data = df.dropna(subset=[x_column, y_column])
        
        if valid_data.empty:
            return {
                "status": "error",
                "error_message": f"No valid numeric data found in columns '{x_column}' and '{y_column}'"
            }
        
        # Set default axis titles if not provided
        if x_axis_title is None:
            x_axis_title = x_column
        if y_axis_title is None:
            y_axis_title = y_column
        
        # Create the figure
        try:
            fig = go.Figure()
        except Exception as e:
            return {
                "status": "error",
                "error_message": f"Error creating figure: {str(e)}"
            }
        
        # Determine colors based on color scheme
        if color_scheme == "plotly":
            colors = px.colors.qualitative.Plotly
            dot_color = colors[0]
        elif color_scheme == "viridis":
            colors = px.colors.sequential.Viridis
            dot_color = colors[len(colors)//2]
        elif color_scheme == "plasma":
            colors = px.colors.sequential.Plasma
            dot_color = colors[len(colors)//2]
        elif color_scheme == "custom":
            dot_color = "#e74c3c"  # Professional red
        else:
            colors = px.colors.qualitative.Set1
            dot_color = colors[0]
        
        # Prepare marker size data
        if size_column:
            # Convert size column to numeric
            size_values = pd.to_numeric(valid_data[size_column], errors='coerce')
            
            # Remove rows with NaN values in size column
            valid_data = valid_data.dropna(subset=[size_column])
            
            if valid_data.empty:
                return {
                    "status": "error",
                    "error_message": f"No valid numeric data found in size column '{size_column}'"
                }
            
            # Normalize size values to the specified range
            min_size, max_size = marker_size_range
            if size_values.min() != size_values.max():
                normalized_sizes = min_size + (size_values - size_values.min()) / (size_values.max() - size_values.min()) * (max_size - min_size)
            else:
                normalized_sizes = [min_size] * len(size_values)
            
            # Create scatter plot with size variation
            fig.add_trace(go.Scatter(
                x=valid_data[x_column],
                y=valid_data[y_column],
                mode='markers',
                name=f"{x_column} vs {y_column}",
                marker=dict(
                    size=normalized_sizes,
                    color=dot_color,
                    opacity=opacity,
                    line=dict(width=1, color='white'),
                    showscale=True,
                    colorscale='Viridis',
                    colorbar=dict(
                        title=dict(
                            text=size_column,
                            font=dict(size=12, color='#2c3e50')
                        ),
                        x=1.02,
                        xanchor='left',
                        len=0.9,
                        thickness=20,
                        outlinewidth=1,
                        outlinecolor='#bdc3c7'
                    )
                ),
                hovertemplate=f'<b>{x_column}:</b> %{{x:.2f}}<br>' +
                             f'<b>{y_column}:</b> %{{y:.2f}}<br>' +
                             f'<b>{size_column}:</b> %{{marker.size:.1f}}<extra></extra>'
            ))
            
        else:
            # Create scatter plot without size variation
            fig.add_trace(go.Scatter(
                x=valid_data[x_column],
                y=valid_data[y_column],
                mode='markers',
                name=f"{x_column} vs {y_column}",
                marker=dict(
                    size=marker_size_range[0],  # Use minimum size
                    color=dot_color,
                    opacity=opacity,
                    line=dict(width=1, color='white')
                ),
                hovertemplate=f'<b>{x_column}:</b> %{{x:.2f}}<br>' +
                             f'<b>{y_column}:</b> %{{y:.2f}}<extra></extra>'
            ))
        
        # Update layout with enhanced styling
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=18, color='#2c3e50', weight='bold')
            ),
            xaxis_title=x_axis_title,
            yaxis_title=y_axis_title,
            showlegend=show_legend,
            height=height,
            width=width,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Arial, sans-serif', size=12, color='#2c3e50'),
            margin=dict(l=60, r=80, t=60, b=60)
        )
        
        # Update axes styling
        fig.update_xaxes(
            gridcolor='#ecf0f1' if show_grid else 'white',
            gridwidth=1 if show_grid else 0,
            zeroline=False,
            showline=True,
            linecolor='#bdc3c7',
            linewidth=1
        )
        
        fig.update_yaxes(
            gridcolor='#ecf0f1' if show_grid else 'white',
            gridwidth=1 if show_grid else 0,
            zeroline=False,
            showline=True,
            linecolor='#bdc3c7',
            linewidth=1
        )
        
        # Export to HTML
        html_content, full_path = self.export_to_html(fig, f"{title}.html")
        
        return {
            "status": "success",
            "html_content": html_content,
            "full_path": full_path
        }
    
    def create_station_map(self, markdown_data: str, 
                          metric_column: Optional[str] = None,
                          data_date: Optional[str] = None,
                          title: str = "Station Map") -> Dict[str, str]:
        """
        Create a dedicated station map visualization tool.
        Reads ALL station coordinates and merges with metric data.
        
        Args:
            markdown_data: Markdown formatted data with two columns. First column: station_code. Second column: metric values.
            metric_column: Column name for the metric values to display (such as 'Daily COH')
            data_date: (Optional) Date of the data (will be appended to title if provided)
            title: Map title
            
        Returns:
            { "status": "success" | "error",  
              "html_content": HTML content as string,
              "full_path": file path where the html chart is saved (string),
              "error_message": "The error message if the status is error" }
        """
        # Load ALL station coordinates from CSV file first
        station_file = os.path.join(project_root, "data", "Station_Locations.csv")
        if not os.path.exists(station_file):
            raise FileNotFoundError(f"Station locations file not found: {station_file}")
        
        station_locations = pd.read_csv(station_file)
        print(f"Loaded {len(station_locations)} stations from location database")
        
        # Filter out specific stations: XNK2 and DSW2
        stations_to_remove = ['XNK2', 'DSW2']
        station_locations = station_locations[~station_locations['Station_Code'].isin(stations_to_remove)]
        print(f"Filtered out stations: {stations_to_remove}")
        print(f"Remaining stations: {len(station_locations)}")
        
        # Update title to include date if provided
        if data_date:
            title = f"{title} ({data_date})"
        
        # Parse the markdown data for metric values
        if markdown_data and markdown_data.strip():
            df = self.parse_markdown_data(markdown_data)
            
            if not df.empty and ('Station_Code' in df.columns or 'station_code' in df.columns):
                # Restore original station codes (parse_markdown_data converts them to numeric)
                import re
                station_code_pattern = r'\|\s*([A-Z]{3}\d+)\s*\|'
                station_codes = re.findall(station_code_pattern, markdown_data)
                
                if len(station_codes) == len(df):
                    df['Station_Code'] = station_codes
                    print(f"Restored {len(station_codes)} station codes from markdown")
                else:
                    print(f"Warning: Could not restore all station codes. Found {len(station_codes)}, expected {len(df)}")
                
                # Filter out specific stations from markdown data as well
                stations_to_remove = ['XNK2', 'DSW2']
                df = df[~df['Station_Code'].isin(stations_to_remove)]
                print(f"Filtered out stations from markdown data: {stations_to_remove}")
                print(f"Remaining stations in markdown data: {len(df)}")
                
                # Ensure station codes are strings for proper merging
                df['Station_Code'] = df['Station_Code'].astype(str)
                station_locations['Station_Code'] = station_locations['Station_Code'].astype(str)
                
                # Merge metric data with station locations (inner join to keep only stations with metric data)
                merged_df = station_locations.merge(df, on='Station_Code', how='inner')
                print(f"Merged data: {len(merged_df)} stations with metric data")
            else:
                # No metric data provided, use all stations with default values
                merged_df = station_locations.copy()
                merged_df['Metric_Value'] = None
                merged_df['Data_Date'] = None
                print("No metric data provided, showing all stations")
        else:
            # No markdown data provided, use all stations with default values
            merged_df = station_locations.copy()
            merged_df['Metric_Value'] = None
            merged_df['Data_Date'] = None
            print("No markdown data provided, showing all stations")
        
        # Filter stations with valid coordinates
        valid_stations = merged_df[
            merged_df['Latitude'].notna() & 
            merged_df['Longitude'].notna()
        ].copy()
        
        if valid_stations.empty:
            raise ValueError("No stations with valid coordinates found")
        
        print(f"Valid stations with coordinates: {len(valid_stations)}")
        
        # Group stations by coordinates for clustering
        print("Grouping stations by coordinates for clustering...")
        station_clusters = self._create_station_clusters(valid_stations)
        
        # Determine which columns to use for visualization
        if metric_column is None:
            # Look for common metric column names (second column after Station_Code)
            if len(valid_stations.columns) > 1:
                # Use the second column as default metric column
                metric_column = valid_stations.columns[1]
                print(f"Auto-detected metric column: {metric_column}")
            else:
                # Use first numeric column as fallback
                numeric_columns = valid_stations.select_dtypes(include=[np.number]).columns.tolist()
                metric_column = numeric_columns[0] if numeric_columns else None
        
        if data_date is None:
            # Look for date column
            date_columns = [col for col in valid_stations.columns if any(keyword in col.lower() for keyword in ['date', 'time', 'created'])]
            if date_columns:
                data_date = date_columns[0]
                print(f"Auto-detected date column: {data_date}")
        
        # Create the map
        try:
            fig = go.Figure()
        except Exception as e:
            return {
                "status": "error",
                "error_message": f"Error creating figure: {str(e)}"
            }
        
        # Create single markers for each unique coordinate location
        #print("Creating single markers for each coordinate location...")
        
        # Track all coordinates and colors for the map
        all_lons = []
        all_lats = []
        all_colors = []
        all_hover_texts = []
        all_marker_sizes = []
        
        # Process each cluster (each cluster = one unique coordinate location)
        for (lat, lon), cluster_stations in station_clusters.items():
            cluster_size = len(cluster_stations)
            
            # Always create ONE marker per coordinate location
            #print(f"  Creating marker for {cluster_size} station(s) at ({lat:.4f}, {lon:.4f})")
            
            # Sort stations within cluster by metric values (largest to smallest)
            if metric_column and metric_column in cluster_stations.columns:
                # Create a copy to avoid modifying the original
                sorted_cluster = cluster_stations.copy()
                # Convert to numeric and sort by metric values (descending)
                sorted_cluster['_temp_metric'] = pd.to_numeric(sorted_cluster[metric_column], errors='coerce').fillna(0)
                sorted_cluster = sorted_cluster.sort_values('_temp_metric', ascending=False)
                # Drop the temporary column
                sorted_cluster = sorted_cluster.drop('_temp_metric', axis=1)
                
                # Calculate maximum metric values for color coding
                metric_values = pd.to_numeric(sorted_cluster[metric_column], errors='coerce').fillna(0)
                max_metric = metric_values.max()
                all_colors.append(max_metric)
                
                print(f"    Stations ordered by {metric_column}: {sorted_cluster[metric_column].tolist()}")
            else:
                sorted_cluster = cluster_stations
                all_colors.append(0)
            
            # Create comprehensive hover text showing ALL stations at this location
            if cluster_size == 1:
                # Single station - simple hover text
                row = sorted_cluster.iloc[0]
                station_code = row['Station_Code']
                metric_value = row.get(metric_column, None) if metric_column else None
                date_value = row.get(data_date, None) if data_date else None
                
                if metric_value is not None and date_value is not None:
                    hover_text = f"{station_code}: {metric_value:.2f} ({date_value})"
                elif metric_value is not None:
                    hover_text = f"{station_code}: {metric_value:.2f}"
                elif date_value is not None:
                    hover_text = f"{station_code}: {date_value}"
                else:
                    # Show station location info instead of "No metric data"
                    city = row.get('City', 'Unknown City')
                    state = row.get('State', 'Unknown State')
                    hover_text = f"{station_code}<br>{city}, {state}<br>Location: ({lat:.4f}, {lon:.4f})"
            else:
                # Multiple stations - show each station's data in the hover (now ordered by metric values)
                hover_parts = []
                for _, row in sorted_cluster.iterrows():
                    station_code = row['Station_Code']
                    metric_value = row.get(metric_column, None) if metric_column else None
                    date_value = row.get(data_date, None) if data_date else None
                    
                    if metric_value is not None and date_value is not None:
                        station_text = f"{station_code}: {metric_value:.2f} ({date_value})"
                    elif metric_value is not None:
                        station_text = f"{station_code}: {metric_value:.2f}"
                    elif date_value is not None:
                        station_text = f"{station_code}: {date_value}"
                    else:
                        # Show station location info instead of "No metric data"
                        city = row.get('City', 'Unknown City')
                        state = row.get('State', 'Unknown State')
                        station_text = f"{station_code}<br>{city}, {state}"
                    
                    hover_parts.append(station_text)
                
                hover_text = "<br>".join(hover_parts)
            
            # Add to lists
            all_lons.append(lon)
            all_lats.append(lat)
            all_hover_texts.append(hover_text)
            
            # Scale marker size based on cluster size (min 8, max 20)
            marker_size = min(8 + (cluster_size - 1) * 2, 20)
            all_marker_sizes.append(marker_size)
        
        # Handle color coding for the entire map
        if metric_column and metric_column in valid_stations.columns:
            colorbar_title = f"{metric_column}"
            show_colorbar = True
            
            # Calculate dynamic color range: 0 to ceil(max(metric))
            max_metric = max(all_colors) if all_colors else 1
            if max_metric > 0:
                cmax = math.ceil(max_metric)
            else:
                cmax = 1  # Fallback if all values are 0
        else:
            # Use uniform color for all stations
            colorbar_title = "Station Locations"
            show_colorbar = False
            cmax = 1  # Fallback for uniform color
        
        # Add the clustered scatter plot
        fig.add_trace(go.Scattergeo(
            lon=all_lons,
            lat=all_lats,
            mode='markers',
            marker=dict(
                size=all_marker_sizes,
                color=all_colors,
                colorscale='RdYlGn_r',  # Red-Yellow-Green reversed (Green=min, Red=max)
                showscale=show_colorbar,
                colorbar=dict(
                    title=dict(
                        text=colorbar_title,
                        font=dict(size=14, color='#2c3e50')
                    ),
                    x=0.75,  # Move colorbar closer to the chart
                    xanchor='left',
                    len=0.9,  # Make colorbar shorter
                    thickness=20,  # Make colorbar thinner
                    outlinewidth=1,
                    outlinecolor='#bdc3c7',
                    tickfont=dict(size=10, color='#2c3e50'),
                    tickformat='.1f'
                ) if show_colorbar else None,
                cmin=0,
                cmax=cmax
            ),
            text=all_hover_texts,
            hovertemplate='%{text}<extra></extra>',
            name="Stations"
        ))
        
        # Update layout for US map with professional styling
        fig.update_geos(
            scope="usa",
            projection_type="albers usa",
            showland=True,
            landcolor="rgb(248, 248, 248)",  # Lighter, more professional land color
            coastlinecolor="rgb(189, 195, 199)",  # Subtle coastline
            showocean=True,
            oceancolor="rgb(236, 240, 241)",  # Professional ocean color
            showlakes=True,
            lakecolor="rgb(255, 255, 255)",
            showrivers=True,
            rivercolor="rgb(189, 195, 199)",  # Subtle river color
            showcountries=False,  # Don't show country borders
            showsubunits=True,  # Show state borders
            subunitcolor="rgb(220, 220, 220)",  # Subtle state border color
            subunitwidth=0.5  # Thin state borders
        )
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,  # Center the title
                xanchor='center',
                y=0.95,  # Position title at the top
                yanchor='top',
                font=dict(
                    size=20,
                    color='#2c3e50',
                    weight='bold',
                    family='Arial, sans-serif'
                )
            ),
            height=700,  # Increase height for better spacing
            margin=dict(l=20, r=80, t=80, b=20),  # Better margins for professional look
            showlegend=False,
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(
                family='Arial, sans-serif',
                size=12,
                color='#2c3e50'
            )
        )
        
        # Export to HTML
        html_content, full_path = self.export_to_html(fig, f"{title}.html")
        
        return {
            "status": "success",
            "html_content": html_content,
            "full_path": full_path
        }
    
    def _apply_coordinate_offsets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply intelligent coordinate offsets to handle duplicate coordinates.
        
        This method adds small offsets to stations with identical coordinates
        to ensure all stations are visible on the map while maintaining
        geographic clustering.
        
        Args:
            df: DataFrame with 'Latitude' and 'Longitude' columns
            
        Returns:
            DataFrame with adjusted coordinates
        """
        if df.empty:
            return df
            
        # Create a copy to avoid modifying the original
        df_adjusted = df.copy()
        
        # Group by coordinates to find duplicates
        coord_groups = df_adjusted.groupby(['Latitude', 'Longitude'])
        
        # Calculate offsets for each group of duplicate coordinates
        for (lat, lon), group in coord_groups:
            if len(group) > 1:
                print(f"  Applying offsets to {len(group)} stations at ({lat}, {lon})")
                
                # Calculate offset pattern based on number of stations
                # Use a spiral pattern to distribute stations around the original point
                for idx, (_, row) in enumerate(group.iterrows()):
                    if idx == 0:
                        # Keep first station at original coordinates
                        continue
                    
                    # Calculate offset using a spiral pattern
                    # Offset increases with distance from center
                    angle = (idx - 1) * (2 * np.pi / (len(group) - 1))
                    distance = 0.001 * (idx ** 0.5)  # Gradual increase in offset
                    
                    # Convert polar coordinates to lat/lon offsets
                    lat_offset = distance * np.cos(angle)
                    lon_offset = distance * np.sin(angle)
                    
                    # Apply offsets
                    df_adjusted.loc[row.name, 'Latitude'] = lat + lat_offset
                    df_adjusted.loc[row.name, 'Longitude'] = lon + lon_offset
                    
                    print(f"    Station {row.get('Station_Code', f'#{idx}')}: offset by ({lat_offset:.6f}, {lon_offset:.6f})")
        
        return df_adjusted
    
    def _create_station_clusters(self, df: pd.DataFrame) -> Dict[Tuple[float, float], pd.DataFrame]:
        """
        Create clusters of stations based on their coordinates.
        
        This method groups stations with identical coordinates into clusters
        for better visualization and user interaction.
        
        Args:
            df: DataFrame with 'Latitude' and 'Longitude' columns
            
        Returns:
            Dictionary mapping coordinate tuples to station groups
        """
        if df.empty:
            return {}
            
        # Group by coordinates to find clusters
        coord_groups = df.groupby(['Latitude', 'Longitude'])
        
        clusters = {}
        for (lat, lon), group in coord_groups:
            clusters[(lat, lon)] = group
            
        print(f"Created {len(clusters)} coordinate clusters")
        
        # Report cluster sizes
        cluster_sizes = [len(group) for group in clusters.values()]
        if cluster_sizes:
            max_cluster_size = max(cluster_sizes)
            total_clustered = sum(size for size in cluster_sizes if size > 1)
            print(f"  Largest cluster: {max_cluster_size} stations")
            print(f"  Total stations in clusters: {total_clustered}")
            
        return clusters
    
    def export_to_html(self, fig: go.Figure, filename: Optional[str] = None) -> Tuple[str, str]:
        """
        Export figure to HTML with CDN loading.
        
        Args:
            fig: Plotly figure object
            filename: Output filename (optional)
            
        Returns:
            Tuple of (HTML content as string, file path as string)
        """
        if filename is None:
            filename = f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(project_root, "outputs", "figures")
        os.makedirs(output_dir, exist_ok=True)
        
        # Construct full file path
        full_path = os.path.join(output_dir, filename)
        
        # Configure for CDN loading
        fig.write_html(full_path, include_plotlyjs='cdn')
        
        # Read and return the HTML content
        with open(full_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        return html_content, full_path
    
    def export_to_png(self, fig: go.Figure, filename: Optional[str] = None) -> str:
        """
        Export figure to PNG.
        
        Args:
            fig: Plotly figure object
            filename: Output filename (optional)
            
        Returns:
            PNG file path
        """
        if filename is None:
            filename = f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        fig.write_image(filename)
        return filename
    
    def get_available_chart_types(self) -> List[str]:
        """Get list of available chart types."""
        return [
            "geographic_map",
            "time_series", 
            "double_y_time_series",
            "correlation_plot",
            "distribution_plot",
            "bar_chart",
            "scatter_plot",
            "station_map"
        ]
    
    def create_visualization(self, markdown_data: str, 
                           chart_type: Optional[str] = None,
                           chart_params: Optional[Dict[str, Any]] = None,
                           output_format: str = "html") -> Union[str, go.Figure]:
        """
        Main function to create visualizations from markdown data.
        
        Args:
            markdown_data: Markdown formatted data
            chart_type: Specific chart type (optional, will auto-detect if None)
            chart_params: Parameters for the specific chart type
            output_format: Output format ("html", "png", or "figure")
            
        Returns:
            HTML string, PNG file path, or Plotly figure object
        """
        # Parse markdown data
        df = self.parse_markdown_data(markdown_data)
        
        if df.empty:
            raise ValueError("No data could be parsed from markdown input")
        
        # Detect data types
        data_info = self.detect_data_type(df)
        
        # Auto-detect chart type if not specified
        if chart_type is None:
            chart_type = self.suggest_chart_type(df, data_info)
        
        # Create the appropriate chart
        if chart_type == "geographic_map":
            if chart_params is None:
                chart_params = {}
            value_column = chart_params.get('value_column', df.select_dtypes(include=[np.number]).columns[0])
            fig = self.create_geographic_map(df, value_column)
            
        elif chart_type == "time_series":
            if chart_params is None:
                chart_params = {}
            value_columns = chart_params.get('value_columns', [])
            title = chart_params.get('title', "Time Series")
            x_axis_title = chart_params.get('x_axis_title', None)
            y_axis_title = chart_params.get('y_axis_title', None)
            show_legend = chart_params.get('show_legend', True)
            line_mode = chart_params.get('line_mode', "lines+markers")
            line_width = chart_params.get('line_width', 2)
            marker_size = chart_params.get('marker_size', 6)
            height = chart_params.get('height', 600)
            width = chart_params.get('width', None)
            color_scheme = chart_params.get('color_scheme', "plotly")
            hover_mode = chart_params.get('hover_mode', "x unified")
            grid_style = chart_params.get('grid_style', "light")
            
            fig = self.create_time_series(
                markdown_data, value_columns, title, x_axis_title, y_axis_title,
                show_legend, line_mode, line_width, marker_size, height, width,
                color_scheme, hover_mode, grid_style
            )
            
        elif chart_type == "double_y_time_series":
            if chart_params is None:
                chart_params = {}
            time_column = chart_params.get('time_column', df.columns[0])
            y1_column = chart_params.get('y1_column', df.select_dtypes(include=[np.number]).columns[0])
            y2_column = chart_params.get('y2_column', df.select_dtypes(include=[np.number]).columns[1])
            fig = self.create_double_y_time_series(df, time_column, y1_column, y2_column)
            
        elif chart_type == "correlation_plot":
            fig = self.create_correlation_plot(df)
            
        elif chart_type == "distribution_plot":
            if chart_params is None:
                chart_params = {}
            markdown_data = chart_params.get('markdown_data', "")
            column = chart_params.get('column', "")
            plot_type = chart_params.get('plot_type', 'histogram')
            title = chart_params.get('title', None)
            x_axis_title = chart_params.get('x_axis_title', None)
            y_axis_title = chart_params.get('y_axis_title', None)
            show_legend = chart_params.get('show_legend', True)
            height = chart_params.get('height', 600)
            width = chart_params.get('width', None)
            color_scheme = chart_params.get('color_scheme', "plotly")
            grid_style = chart_params.get('grid_style', "light")
            bins = chart_params.get('bins', None)
            opacity = chart_params.get('opacity', 0.7)
            
            result = self.create_distribution_plot(
                markdown_data, column, plot_type, title, x_axis_title, y_axis_title,
                show_legend, height, width, color_scheme, grid_style, bins, opacity
            )
            
            if result["status"] == "success":
                fig = result["figure"]
            else:
                raise ValueError(result["error_message"])
            
        elif chart_type == "bar_chart":
            if chart_params is None:
                chart_params = {}
            markdown_data = chart_params.get('markdown_data', "")
            x_column = chart_params.get('x_column', "")
            y_column = chart_params.get('y_column', "")
            title = chart_params.get('title', "Bar Chart")
            x_axis_title = chart_params.get('x_axis_title', None)
            y_axis_title = chart_params.get('y_axis_title', None)
            show_legend = chart_params.get('show_legend', True)
            height = chart_params.get('height', 600)
            width = chart_params.get('width', None)
            color_scheme = chart_params.get('color_scheme', "plotly")
            orientation = chart_params.get('orientation', "v")
            opacity = chart_params.get('opacity', 0.8)
            text_auto = chart_params.get('text_auto', "auto")
            show_text = chart_params.get('show_text', False)
            
            result = self.create_bar_chart(
                markdown_data, x_column, y_column, title, x_axis_title, y_axis_title,
                show_legend, height, width, color_scheme, orientation, opacity, text_auto, show_text
            )
            
            if result["status"] == "success":
                fig = result["figure"]
            else:
                raise ValueError(result["error_message"])
            
        elif chart_type == "scatter_plot":
            if chart_params is None:
                chart_params = {}
            markdown_data = chart_params.get('markdown_data', "")
            x_column = chart_params.get('x_column', "")
            y_column = chart_params.get('y_column', "")
            size_column = chart_params.get('size_column', None)
            title = chart_params.get('title', "Scatter Plot")
            x_axis_title = chart_params.get('x_axis_title', None)
            y_axis_title = chart_params.get('y_axis_title', None)
            show_legend = chart_params.get('show_legend', True)
            height = chart_params.get('height', 600)
            width = chart_params.get('width', None)
            color_scheme = chart_params.get('color_scheme', "plotly")
            marker_size_range = chart_params.get('marker_size_range', (8, 20))
            opacity = chart_params.get('opacity', 0.7)
            show_grid = chart_params.get('show_grid', True)
            
            result = self.create_scatter_plot(
                markdown_data, x_column, y_column, size_column, title, x_axis_title, y_axis_title,
                show_legend, height, width, color_scheme, marker_size_range, opacity, show_grid
            )
            
            if result["status"] == "success":
                fig = result["figure"]
            else:
                raise ValueError(result["error_message"])
            
        elif chart_type == "station_map":
            if chart_params is None:
                chart_params = {}
            markdown_data = chart_params.get('markdown_data', "")
            metric_column = chart_params.get('metric_column', None)
            data_date = chart_params.get('data_date', None)
            title = chart_params.get('title', "Station Map")
            fig = self.create_station_map(markdown_data, metric_column, data_date, title)
            
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        
        # Return in requested format
        if output_format == "html":
            html_content, _ = self.export_to_html(fig)
            return html_content
        elif output_format == "png":
            return self.export_to_png(fig)
        elif output_format == "figure":
            return fig
        else:
            raise ValueError(f"Unsupported output format: {output_format}")


# Convenience functions for direct use
def create_visualization(markdown_data: str, 
                        chart_type: Optional[str] = None,
                        chart_params: Optional[Dict[str, Any]] = None,
                        output_format: str = "html") -> Union[str, go.Figure]:
    """
    Convenience function to create visualizations without instantiating the class.
    
    Args:
        markdown_data: Markdown formatted data
        chart_type: Specific chart type (optional)
        chart_params: Parameters for the specific chart type
        output_format: Output format ("html", "png", or "figure")
        
    Returns:
        HTML string, PNG file path, or Plotly figure object
    """
    visualizer = EnhancedDataVisualizer()
    return visualizer.create_visualization(markdown_data, chart_type, chart_params, output_format)


def create_station_map_visualization(markdown_data: str,
                                   metric_column: Optional[str] = None,
                                   data_date: Optional[str] = None,
                                   title: str = "Station Map",
                                   output_format: str = "html") -> Union[str, go.Figure]:
    """
    Convenience function to create station map visualizations.
    
    Args:
        markdown_data: Markdown formatted data with station codes and metric values
        metric_column: Column name for the metric values to display (defaults to first numeric column)
        data_date: Date of the data (optional, will be appended to title if provided)
        title: Map title
        output_format: Output format ("html", "png", or "figure")
        
    Returns:
        HTML string, PNG file path, or Plotly figure object
    """
    visualizer = EnhancedDataVisualizer()
    fig = visualizer.create_station_map(markdown_data, metric_column, data_date, title)
    
    # Return in requested format
    if output_format == "html":
        return visualizer.export_to_html(fig)
    elif output_format == "png":
        return visualizer.export_to_png(fig)
    elif output_format == "figure":
        return fig
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def create_time_series_visualization(markdown_data: str,
                                   value_columns: list,
                                   title: str = "Time Series",
                                   x_axis_title: Optional[str] = None,
                                   y_axis_title: Optional[str] = None,
                                   show_legend: bool = True,
                                   line_mode: str = "lines+markers",
                                   line_width: int = 2,
                                   marker_size: int = 6,
                                   height: int = 600,
                                   width: Optional[int] = None,
                                   color_scheme: str = "plotly",
                                   hover_mode: str = "x unified",
                                   grid_style: str = "light",
                                   output_format: str = "html") -> Union[str, go.Figure]:
    """
    Convenience function to create time series visualizations.
    
    Args:
        markdown_data: Markdown formatted data where the first column is datetime (YYYY-MM-DD format)
        value_columns: List of column names (strings) containing numeric values to plot
        title: Chart title
        x_axis_title: Custom title for x-axis (defaults to "Date" if None)
        y_axis_title: Custom title for y-axis (defaults to "Value" if None)
        show_legend: Whether to show the legend
        line_mode: Plot mode - "lines", "markers", or "lines+markers"
        line_width: Width of the lines
        marker_size: Size of the markers
        height: Chart height in pixels
        width: Chart width in pixels (None for auto)
        color_scheme: Color scheme for the lines ("plotly", "viridis", "plasma", etc.)
        hover_mode: Hover mode - "x unified", "y unified", "closest", or "x"
        grid_style: Grid style - "light", "dark", "white", or "none"
        output_format: Output format ("html", "png", or "figure")
        
    Returns:
        HTML string, PNG file path, or Plotly figure object
    """
    visualizer = DataVisualizer()
    fig = visualizer.create_time_series(
        markdown_data, value_columns, title, x_axis_title, y_axis_title,
        show_legend, line_mode, line_width, marker_size, height, width,
        color_scheme, hover_mode, grid_style
    )
    
    # Return in requested format
    if output_format == "html":
        return visualizer.export_to_html(fig)
    elif output_format == "png":
        return visualizer.export_to_png(fig)
    elif output_format == "figure":
        return fig
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def create_distribution_plot_visualization(markdown_data: str,
                                         column: str,
                                         plot_type: str = "histogram",
                                         title: Optional[str] = None,
                                         x_axis_title: Optional[str] = None,
                                         y_axis_title: Optional[str] = None,
                                         show_legend: bool = True,
                                         height: int = 600,
                                         width: Optional[int] = None,
                                         color_scheme: str = "plotly",
                                         grid_style: str = "light",
                                         bins: Optional[int] = None,
                                         opacity: float = 0.7,
                                         output_format: str = "html") -> Union[str, go.Figure]:
    """
    Convenience function to create distribution plot visualizations.
    
    Args:
        markdown_data: Markdown formatted data where the first column is typically the metric to plot
        column: Column name (string) containing the numeric values to visualize
        plot_type: Type of distribution plot - "histogram", "box", or "violin" (string)
        title: Chart title (string, defaults to auto-generated title if None)
        x_axis_title: Custom title for x-axis (string, defaults to column name if None)
        y_axis_title: Custom title for y-axis (string, defaults to "Count" or "Value" if None)
        show_legend: Whether to show the legend (boolean)
        height: Chart height in pixels (integer)
        width: Chart width in pixels (None for auto)
        color_scheme: Color scheme for the plot (string)
        grid_style: Grid style - "light", "dark", "white", or "none" (string)
        bins: Number of bins for histogram (integer, only applies to histogram)
        opacity: Opacity of the plot elements (float, 0.0 to 1.0)
        output_format: Output format ("html", "png", or "figure")
        
    Returns:
        HTML string, PNG file path, or Plotly figure object
    """
    visualizer = DataVisualizer()
    result = visualizer.create_distribution_plot(
        markdown_data, column, plot_type, title, x_axis_title, y_axis_title,
        show_legend, height, width, color_scheme, grid_style, bins, opacity
    )
    
    if result["status"] == "success":
        fig = result["figure"]
    else:
        raise ValueError(result["error_message"])
    
    # Return in requested format
    if output_format == "html":
        return visualizer.export_to_html(fig)
    elif output_format == "png":
        return visualizer.export_to_png(fig)
    elif output_format == "figure":
        return fig
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def create_bar_chart_visualization(markdown_data: str,
                                 x_column: str,
                                 y_column: str,
                                 title: str = "Bar Chart",
                                 x_axis_title: Optional[str] = None,
                                 y_axis_title: Optional[str] = None,
                                 show_legend: bool = True,
                                 height: int = 600,
                                 width: Optional[int] = None,
                                 color_scheme: str = "plotly",
                                 orientation: str = "v",
                                 opacity: float = 0.8,
                                 text_auto: str = "auto",
                                 show_text: bool = False,
                                 output_format: str = "html") -> Union[str, go.Figure]:
    """
    Convenience function to create bar chart visualizations.
    
    Args:
        markdown_data: Markdown formatted data with categorical and numeric columns
        x_column: Column name (string) for x-axis (categories)
        y_column: Column name (string) for y-axis (numeric values)
        title: Chart title (string)
        x_axis_title: Custom title for x-axis (string, defaults to x_column if None)
        y_axis_title: Custom title for y-axis (string, defaults to y_column if None)
        show_legend: Whether to show the legend (boolean)
        height: Chart height in pixels (integer)
        width: Chart width in pixels (None for auto)
        color_scheme: Color scheme for the bars (string)
        orientation: Bar orientation - "v" for vertical, "h" for horizontal (string)
        opacity: Opacity of the bars (float, 0.0 to 1.0)
        text_auto: Automatically decide position of text labels on bars (boolean)
        show_text: Whether to show value labels on bars (boolean)
        output_format: Output format ("html", "png", or "figure")
        
    Returns:
        HTML string, PNG file path, or Plotly figure object
    """
    visualizer = DataVisualizer()
    result = visualizer.create_bar_chart(
        markdown_data, x_column, y_column, title, x_axis_title, y_axis_title,
        show_legend, height, width, color_scheme, orientation, opacity, text_auto, show_text
    )
    
    if result["status"] == "success":
        fig = result["figure"]
    else:
        raise ValueError(result["error_message"])
    
    # Return in requested format
    if output_format == "html":
        return visualizer.export_to_html(fig)
    elif output_format == "png":
        return visualizer.export_to_png(fig)
    elif output_format == "figure":
        return fig
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def create_scatter_plot_visualization(markdown_data: str,
                                    x_column: str,
                                    y_column: str,
                                    size_column: Optional[str] = None,
                                    title: str = "Scatter Plot",
                                    x_axis_title: Optional[str] = None,
                                    y_axis_title: Optional[str] = None,
                                    show_legend: bool = True,
                                    height: int = 600,
                                    width: Optional[int] = None,
                                    color_scheme: str = "plotly",
                                    marker_size_range: List[int] = [8, 20],
                                    opacity: float = 0.7,
                                    show_grid: bool = True,
                                    output_format: str = "html") -> Union[str, go.Figure]:
    """
    Convenience function to create scatter plot visualizations.
    
    Args:
        markdown_data: Markdown formatted data with numeric columns
        x_column: Column name (string) for x-axis coordinates
        y_column: Column name (string) for y-axis coordinates
        size_column: Column name (string) for dot sizes (optional, defaults to None)
        title: Chart title (string)
        x_axis_title: Custom title for x-axis (string, defaults to x_column if None)
        y_axis_title: Custom title for y-axis (string, defaults to y_column if None)
        show_legend: Whether to show the legend (boolean)
        height: Chart height in pixels (integer)
        width: Chart width in pixels (None for auto)
        color_scheme: Color scheme for the dots (string)
        marker_size_range: Tuple of (min_size, max_size) for dot sizing (tuple)
        opacity: Opacity of the dots (float, 0.0 to 1.0)
        show_grid: Whether to show grid lines (boolean)
        output_format: Output format ("html", "png", or "figure")
        
    Returns:
        HTML string, PNG file path, or Plotly figure object
    """
    visualizer = DataVisualizer()
    result = visualizer.create_scatter_plot(
        markdown_data, x_column, y_column, size_column, title, x_axis_title, y_axis_title,
        show_legend, height, width, color_scheme, marker_size_range, opacity, show_grid
    )
    
    if result["status"] == "success":
        fig = result["figure"]
    else:
        raise ValueError(result["error_message"])
    
    # Return in requested format
    if output_format == "html":
        return visualizer.export_to_html(fig)
    elif output_format == "png":
        return visualizer.export_to_png(fig)
    elif output_format == "figure":
        return fig
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def get_available_chart_types() -> List[str]:
    """Get list of available chart types."""
    return [
        "geographic_map",
        "time_series", 
        "double_y_time_series",
        "correlation_plot",
        "distribution_plot",
        "bar_chart",
        "scatter_plot",
        "station_map"
    ]


if __name__ == "__main__":
    # Example usage
    visualizer = EnhancedDataVisualizer()
    
    # Test with sample data
    sample_markdown = """
    | Station_Code | Metric_Value | Date |
    |--------------|--------------|------|
    | ST001 | 85.5 | 2024-01-01 |
    | ST002 | 92.3 | 2024-01-01 |
    | ST003 | 78.9 | 2024-01-01 |
    """
    
    try:
        result = visualizer.create_visualization(sample_markdown, output_format="html")
        print("Visualization created successfully!")
        print(f"Output length: {len(result)} characters")
    except Exception as e:
        print(f"Error: {e}")
