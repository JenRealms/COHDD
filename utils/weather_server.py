import os
import requests
from typing import Tuple, Optional, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
load_dotenv(override=True)

api_key = os.getenv("OPENWEATHER_API_KEY")


def date_to_unix_timestamp(date_str: str) -> Optional[int]:
    """
    Convert a date string in MM-DD-YYYY format to Unix timestamp (UTC timezone).
    
    Args:
        date_str (str): Date string in MM-DD-YYYY format (e.g., "04-10-2020")
        
    Returns:
        Optional[int]: Unix timestamp (seconds since epoch) or None if invalid format
        
    Examples:
        >>> date_to_unix_timestamp("04-10-2020")
        1586468027
        >>> date_to_unix_timestamp("12-25-2023")
        1703462400
    """
    try:
        # Parse the date string in MM-DD-YYYY format
        date_obj = datetime.strptime(date_str, "%m-%d-%Y")
        
        # Convert to Unix timestamp (UTC timezone)
        unix_timestamp = int(date_obj.timestamp())
        
        return unix_timestamp
        
    except ValueError as e:
        print(f"Invalid date format: {date_str}. Expected format: MM-DD-YYYY")
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Error converting date to Unix timestamp: {e}")
        return None


def unix_timestamp_to_date(unix_timestamp: int) -> Optional[str]:
    """
    Convert a Unix timestamp to MM-DD-YYYY format.
    
    Args:
        unix_timestamp (int): Unix timestamp (seconds since epoch)
        
    Returns:
        Optional[str]: Date string in MM-DD-YYYY format or None if invalid
        
    Examples:
        >>> unix_timestamp_to_date(1586468027)
        "04-10-2020"
    """
    try:
        # Convert Unix timestamp to datetime object
        date_obj = datetime.fromtimestamp(unix_timestamp)
        
        # Format as MM-DD-YYYY
        date_str = date_obj.strftime("%m-%d-%Y")
        
        return date_str
        
    except (ValueError, OSError) as e:
        print(f"Invalid Unix timestamp: {unix_timestamp}")
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Error converting Unix timestamp to date: {e}")
        return None

def get_city_coordinates(city: str, state: Optional[str] = None, country: Optional[str] = "US") -> Optional[Tuple[float, float]]:
    """
    Get the latitude and longitude coordinates for a given city.
    
    Args:
        city (str): The name of the city to get coordinates for
        state (Optional[str]): The state/province code (e.g., "IL", "CA", "TX") for disambiguation
        country (Optional[str]): The country code (default: "US")
        
    Returns:
        Optional[Tuple[float, float]]: A tuple of (latitude, longitude) in decimal degrees,
                                      or None if the city is not found or an error occurs
        
    Examples:
        >>> get_city_coordinates("Springfield")  # Returns most relevant Springfield
        >>> get_city_coordinates("Springfield", "IL")  # Springfield, Illinois specifically
        >>> get_city_coordinates("Springfield", "MA")  # Springfield, Massachusetts specifically
        >>> get_city_coordinates("Portland", "OR")  # Portland, Oregon
        
    Raises:
        ValueError: If the API key is not configured
    """
    if not api_key:
        raise ValueError("OPENWEATHER_API_KEY environment variable is not set")
    
    # OpenWeatherMap Geocoding API endpoint
    base_url = "http://api.openweathermap.org/geo/1.0/direct"
    
    # Build the query string based on available parameters
    if state and country:
        query = f"{city},{state},{country}"
    elif state:
        query = f"{city},{state}"
    else:
        query = city
    
    # Parameters for the API request
    params = {
        "q": query,
        "limit": 1,  # Get only the first (most relevant) result
        "appid": api_key,
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        data = response.json()
        
        if not data:
            location_str = f"{city}, {state}" if state else city
            print(f"Location '{location_str}' not found")
            return None
        
        # Extract coordinates from the first result
        location = data[0]
        lat = location.get("lat")
        lon = location.get("lon")
        
        if lat is None or lon is None:
            location_str = f"{city}, {state}" if state else city
            print(f"Could not extract coordinates for '{location_str}'")
            return None
        
        # Validate coordinate ranges
        if not (-90 <= lat <= 90):
            location_str = f"{city}, {state}" if state else city
            print(f"Invalid latitude value for '{location_str}': {lat}")
            return None
        
        if not (-180 <= lon <= 180):
            location_str = f"{city}, {state}" if state else city
            print(f"Invalid longitude value for '{location_str}': {lon}")
            return None
        
        return (lat, lon)
        
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        return None
    except (KeyError, IndexError, ValueError) as e:
        print(f"Error parsing API response: {e}")
        return None

def get_weather(city: str, date: str, state: Optional[str] = None) -> Dict[str, str]:
    """
    Get the weather report for a given city and date.

    Args:
        city (str): The name of the city to get weather for
        date (str): The date to get weather for in MM-DD-YYYY format (e.g. 08-21-2025)
        state (Optional[str]): The state/province code (e.g., "IL", "CA", "TX") for disambiguation
        
    Returns:
        {
            "status": "success" | "error",
            "weather_report": "The weather report for the given city and date",
            "error_message": "The error message if the status is error"
        }
        
    Examples:
        >>> get_weather("Springfield", "08-21-2025")  # Most relevant Springfield
        >>> get_weather("Springfield", "08-21-2025", "IL")  # Springfield, Illinois
        >>> get_weather("Portland", "08-21-2025", "OR")  # Portland, Oregon
    """
    if not api_key:
        raise ValueError("OPENWEATHER_API_KEY environment variable is not set")
    
    # Get city coordinates
    coordinates = get_city_coordinates(city, state)
    if coordinates is None:
        location_str = f"{city}, {state}" if state else city
        print(f"Could not get coordinates for location: {location_str}")
        return None
    
    lat, lon = coordinates
    
    # Convert date to Unix timestamp
    unix_timestamp = date_to_unix_timestamp(date)
    if unix_timestamp is None:
        print(f"Invalid date format: {date}")
        return None

    # OpenWeatherMap One Call API 3.0 endpoint for historical data
    base_url = "https://api.openweathermap.org/data/3.0/onecall/timemachine"
    
    params = {
        "lat": lat,
        "lon": lon,
        "dt": unix_timestamp,
        "units": "imperial",  # Use metric units for Celsius
        "appid": api_key
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Check if we have data
        if not data or "data" not in data or not data["data"]:
            location_str = f"{city}, {state}" if state else city
            print(f"No weather data found for {location_str} on {date}")
            return None
        
        # Get the first (and usually only) data point
        weather_data = data["data"][0]

        #print("*" * 50)
        #print(weather_data)
        #print("*" * 50)
        
        # Extract weather information
        temp = weather_data.get("temp", "N/A")
        feels_like = weather_data.get("feels_like", "N/A")
        humidity = weather_data.get("humidity", "N/A")
        wind_speed = weather_data.get("wind_speed", "N/A")
        visibility = weather_data.get("visibility", "N/A")
        
        # Get weather description
        weather_info = weather_data.get("weather", [{}])
        description = weather_info[0].get("description", "Unknown") if weather_info else "Unknown"
        
        # Format the date for display
        display_date = unix_timestamp_to_date(weather_data.get("dt", unix_timestamp))
        
        # Create formatted weather report
        location_str = f"{city}, {state}" if state else city
        weather_report = (
            f"Location: {location_str}\n"
            f"Date: {display_date}\n"
            f"Temperature: {temp}°F\n"
            f"Feels Like: {feels_like}°F\n"
            f"Humidity: {humidity}%\n"
            f"Wind Speed: {wind_speed} m/s\n"
            f"Visibility: {visibility} m\n"
            f"Weather: {description}\n"
        )
        
        return {"status": "success", 
                "weather_report": weather_report}
    
    except requests.exceptions.RequestException as e:
        #print(f"Error making API request: {e}")
        return {"status": "error", "error_message": f"Error making API request: {e}"}
    except (KeyError, IndexError, ValueError) as e:
        #print(f"Error parsing API response: {e}")
        return {"status": "error", "error_message": f"Error parsing API response: {e}"}

#city = 'NYC'
#date = '08-09-2025'

#weather_report = get_weather(city, date)
#print(f"Weather report for {city} on {date}:")
#print(weather_report)