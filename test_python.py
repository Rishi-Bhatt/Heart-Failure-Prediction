print("Python is working!")

# Try to import the essential packages
try:
    import flask
    print("Flask is installed!")
except ImportError:
    print("Flask is not installed.")

try:
    import numpy
    print("NumPy is installed!")
except ImportError:
    print("NumPy is not installed.")

try:
    import pandas
    print("Pandas is installed!")
except ImportError:
    print("Pandas is not installed.")

try:
    import sklearn
    print("Scikit-learn is installed!")
except ImportError:
    print("Scikit-learn is not installed.")

try:
    import joblib
    print("Joblib is installed!")
except ImportError:
    print("Joblib is not installed.")
