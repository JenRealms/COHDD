# COHDD - Capped Out Hours Deep Dive Analysis Platform

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

COHDD is an intelligent analytics platform designed for last-mile delivery operations, specializing in Capped Out Hours (COH) analysis and deep dive investigations. The platform leverages multiple AI agents to provide comprehensive insights into delivery capacity constraints, root cause analysis, and operational optimization.

## 🚀 Features

- **Multi-Agent AI System**: Coordinated agents for specialized analysis tasks
- **Real-time Data Analysis**: AWS Athena integration for live COH data
- **Advanced Visualization**: Interactive charts and maps for operational insights
- **Weather Integration**: OpenWeather API for environmental impact analysis
- **Root Cause Analysis**: Automated attribution and constraint identification
- **Comprehensive Reporting**: Deep dive reports with actionable recommendations

## 📋 Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- AWS account with appropriate permissions
- OpenWeather API key

## 🛠️ Installation

### 1. Install uv Package Manager

First, install the `uv` package manager if you haven't already:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using pip
pip install uv
```

### 2. Clone and Setup Project

```bash
# Clone the repository
git clone <your-repo-url>
cd COHDD

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync
```

### 3. Environment Configuration

```bash
# Copy environment template
cp .env-example .env

# Edit .env with your credentials
# Required variables:
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - AWS_DEFAULT_REGION
# - AWS_Model_ID
# - OPENWEATHER_API_KEY
# - TAVILY_API_KEY
```

## 🚀 Getting Started

### Launch the Web Interface

```bash
# Activate virtual environment (if not already active)
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Launch the ADK web interface
adk web
```

The web interface will be available at `http://localhost:8000` (or the port specified in your configuration).

## 🤖 AI Agents Overview

### COH Agent - Main Coordinator

The **COH Agent** is your primary interface for conducting comprehensive COH analysis. It coordinates multiple specialized agents to provide complete insights.

#### Key Features:
- **Multi-Agent Coordination**: Orchestrates weather, RCA, web search, data collection, and visualization agents
- **Intelligent Workflow Planning**: Automatically detects user intent and creates step-by-step analysis plans
- **Context-Aware Analysis**: Integrates weather data, station information, and operational context
- **Comprehensive Reporting**: Combines multiple data sources for complete insights

#### Capabilities:
- Root Cause Attribution (RCA) analysis
- Weather impact assessment
- Station location and regional analysis
- Data collection and visualization
- File management and report generation

### DD Writer Agent - Deep Dive Reports

The **DD Writer Agent** specializes in creating comprehensive Deep Dive (DD) reports by analyzing multiple COH factors in parallel.

#### Key Features:
- **Parallel Analysis**: Runs 10 specialized analysis agents simultaneously
- **Multi-Factor Investigation**: Analyzes backlog, capacity changes, demand signals, exclusions, flex operations, mechanical issues, and weather impacts
- **Professional Report Generation**: Synthesizes findings into actionable insights
- **Data Visualization**: Creates charts and graphs to support analysis

#### Analysis Areas:
1. **Backlog Impact Analysis**: Evaluates backlog management effectiveness
2. **Capacity Change Analysis**: Assesses capacity planning and adjustments
3. **Demand Signal Analysis**: Examines demand forecasting accuracy
4. **Exclusions Impact**: Analyzes exclusion policies and effects
5. **Flex Operations**: Evaluates UTR/OTR flex-up capabilities
6. **Mechanical Issues**: Identifies equipment-related constraints
7. **Weather Impact**: Assesses environmental factors
8. **Root Cause Attribution**: Provides primary constraint identification

## 💡 Use Case Examples

### COH Agent Use Cases

#### 1. Create COH Distribution Map for Today
```
"Create a map showing COH distribution across all US stations for today"
```
**What it does:**
- Collects current COH data for all stations
- Creates an interactive geographic visualization
- Shows COH hotspots and regional patterns
- Provides station-level detail on hover

#### 2. Root Cause Attribution for Specific Station
```
"Analyze the root cause attribution of COH for station DML6 on 2025-08-13"
```
**What it does:**
- Conducts comprehensive RCA analysis
- Integrates weather data for context
- Identifies primary constraints and contributing factors
- Provides actionable recommendations

#### 3. Weather Impact Assessment
```
"Assess weather impact on station DHI2 operations for 2025-02-14"
```
**What it does:**
- Retrieves historical weather data
- Analyzes weather-related capacity changes
- Correlates weather events with COH patterns
- Provides weather mitigation strategies

### DD Writer Agent Use Cases

#### 1. Comprehensive Deep Dive Report
```
"Create a deep dive report for station DGE4 on 2025-08-05"
```
**What it does:**
- Runs parallel analysis across all 10 factors
- Collects comprehensive data for the 30-day period
- Generates professional report with executive summary
- Includes data visualizations and recommendations
- Saves report in both Markdown and HTML formats

#### 2. Multi-Station Comparative Analysis
```
"Generate deep dive analysis comparing stations DBM3, DML6, and DNA6 for 2025-08-13"
```
**What it does:**
- Analyzes each station individually
- Identifies common patterns and unique challenges
- Provides comparative insights
- Recommends best practices and optimization strategies
- Do not include more than 3 stations

**IMPORTANT**
Create a New Sesstion whenever you create a new report.

## 🔧 Configuration

### Environment Variables

The following environment variables are required:

```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_DEFAULT_REGION=us-east-1
AWS_Model_ID=your_aws_model_id

# External APIs
OPENWEATHER_API_KEY=your_openweather_api_key
TAVILY_API_KEY=your_tavily_api_key

# Optional Configuration
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO
```

### AWS Services

- **S3**: Data storage and retrieval
- **Athena**: SQL query execution for COH data
- **IAM**: Access control and permissions

## 📊 Data Sources

The platform integrates with multiple data sources:

- **COH Database**: Primary operational data (station_code, ofd_date, capped_out_hours, etc.)
- **Weather Data**: OpenWeather API for environmental context
- **Web Search Information**: Geographic data via web search 
- **Capacity Metrics**: UTR, OTR, mechanical, and constraint data

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.




---

**Note**: This platform is designed for operational use in last-mile delivery operations. Ensure proper data access permissions and follow your organization's security policies when deploying in production environments.
