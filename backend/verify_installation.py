import sys
import importlib

# List of packages to verify
packages = [
    'flask',
    'flask_cors',
    'numpy',
    'pandas',
    'sklearn',
    'xgboost',
    'shap',
    'neurokit2',
    'matplotlib',
    'joblib'
]

print("Verifying package installations:")
print("-" * 30)

all_installed = True

for package in packages:
    try:
        # Try to import the package
        module = importlib.import_module(package)
        # Get the version if available
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package} (version: {version})")
    except ImportError:
        print(f"❌ {package} - NOT INSTALLED")
        all_installed = False

print("-" * 30)
if all_installed:
    print("All required packages are installed! 🎉")
else:
    print("Some packages are missing. Please install them using:")
    print("pip install -r requirements.txt")

print("\nPython version:", sys.version)
