# Alternative Installation Methods for Heart Failure Prediction System

If you're experiencing issues with the standard installation process, this guide provides several alternative methods to set up the Heart Failure Prediction System.

## Method 1: Step-by-Step Installation Script

This script installs packages one by one with error handling, which can help identify and work around problematic packages.

```bash
# Make the script executable
chmod +x install_dependencies.sh

# Run the script
./install_dependencies.sh
```

The script will:
- Install essential packages one by one
- Try alternative installation methods if a package fails
- Ask if you want to install optional packages

## Method 2: Alternative Requirements Files

We provide two alternative requirements files with different levels of constraints:

### Compatible Requirements

This file has relaxed version constraints that may work better with your system:

```bash
pip install -r backend/requirements_compatible.txt
```

### Minimal Requirements

This file includes only the essential packages needed to run a basic version of the system:

```bash
pip install -r backend/requirements_minimal.txt
```

## Method 3: Conda Environment

Conda provides more reliable package management than pip for scientific Python packages:

```bash
# Make the script executable
chmod +x setup_conda_env.sh

# Run the script
./setup_conda_env.sh
```

The script will:
- Create a new conda environment with Python 3.8
- Install essential packages using conda
- Install optional packages if requested
- Create necessary directories

## Method 4: Minimal Version

We provide a minimal version of the application that doesn't rely on problematic dependencies:

```bash
# Make the script executable
chmod +x run_minimal.sh

# Run the script
./run_minimal.sh
```

The script will:
- Check if Flask is installed and install it if needed
- Start a simplified backend server
- Check if Node.js is installed
- Install frontend dependencies if needed
- Start the frontend development server

### Features of the Minimal Version

The minimal version includes:
- Basic patient data input
- Simplified ECG generation (without NeuroKit2)
- Simple risk prediction algorithm
- Basic visualization of results
- Local storage of patient records

Some advanced features like detailed ECG analysis and SHAP explainability are simplified in this version.

## Method 5: Manual Installation

If all else fails, you can try installing packages manually:

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages one by one
pip install flask
pip install flask-cors
pip install numpy
pip install pandas
pip install scikit-learn
pip install joblib

# Try installing optional packages
pip install matplotlib
pip install xgboost
pip install shap
pip install neurokit2
```

If a specific package fails, try installing with the `--no-deps` flag:

```bash
pip install --no-deps package_name
```

## Troubleshooting

If you continue to experience issues, please refer to the `TROUBLESHOOTING.md` file for more detailed guidance on resolving common problems.
