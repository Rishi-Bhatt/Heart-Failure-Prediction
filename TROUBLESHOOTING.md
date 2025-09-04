# Troubleshooting Guide for Heart Failure Prediction System

This guide addresses common issues you might encounter when setting up and running the Heart Failure Prediction System.

## Backend Issues

### Python Package Installation Problems

**Issue**: Error installing Python packages from requirements.txt or terminal crashes

**Solution**:

#### Option 1: Use the step-by-step installation script

Run the provided installation script that installs packages one by one with error handling:

```bash
./install_dependencies.sh
```

#### Option 2: Use alternative requirements files

Try using the compatible or minimal requirements files:

```bash
# For a more compatible version with relaxed constraints
pip install -r backend/requirements_compatible.txt

# For minimal requirements (only essential packages)
pip install -r backend/requirements_minimal.txt
```

#### Option 3: Use Conda environment

Conda provides more reliable package management:

```bash
./setup_conda_env.sh
```

#### Option 4: Run the minimal version

Use the minimal version of the application that doesn't rely on problematic dependencies:

```bash
./run_minimal.sh
```

#### Option 5: Manual installation

Install packages one by one to identify the problematic package:

```bash
pip install flask
pip install flask-cors
pip install numpy
# ... and so on
```

If a specific package fails, try installing with the `--no-deps` flag:

```bash
pip install --no-deps package_name
```

### Flask Server Won't Start

**Issue**: The Flask server doesn't start or immediately crashes

**Solution**:

1. Check for port conflicts:

   ```bash
   # On macOS/Linux
   lsof -i :5000
   # On Windows
   netstat -ano | findstr :5000
   ```

   If another process is using port 5000, either terminate that process or change the port in `app.py`.

2. Check for import errors by running:

   ```bash
   cd backend
   python -c "from utils.ecg_generator import generate_ecg; print('Import successful')"
   ```

3. Make sure all `__init__.py` files exist in the package directories.

### Model Training Issues

**Issue**: The model training process hangs or crashes

**Solution**:

1. Try running with a smaller dataset by modifying `n_samples` in `heart_failure_model.py`.

2. Check if you have enough disk space for model storage.

3. Try using the simplified model for testing:
   ```bash
   cd backend
   python test_simple_model.py
   ```

## Frontend Issues

### npm Install Errors

**Issue**: Errors during `npm install`

**Solution**:

1. Make sure you're using a compatible Node.js version (14+):

   ```bash
   node --version
   ```

2. Clear npm cache:

   ```bash
   npm cache clean --force
   ```

3. Delete node_modules and package-lock.json and try again:
   ```bash
   rm -rf node_modules package-lock.json
   npm install
   ```

### Vite Development Server Issues

**Issue**: Vite development server won't start

**Solution**:

1. Check for port conflicts:

   ```bash
   # On macOS/Linux
   lsof -i :5173
   # On Windows
   netstat -ano | findstr :5173
   ```

2. Try running with a different port:

   ```bash
   npm run dev -- --port 3000
   ```

3. Check for ESLint configuration issues:
   ```bash
   npx eslint --init
   ```

### API Connection Issues

**Issue**: Frontend can't connect to backend API

**Solution**:

1. Make sure the backend server is running.

2. Check that the API URL in the frontend code matches the backend server address.

3. Verify that CORS is properly configured in the backend.

4. Try testing the API with curl:
   ```bash
   curl -X GET http://localhost:5000/api/test
   ```

## Data Storage Issues

**Issue**: Can't save or load patient data

**Solution**:

1. Make sure the data directories exist:

   ```bash
   mkdir -p backend/data/patients
   mkdir -p backend/models
   ```

2. Check file permissions:

   ```bash
   # On macOS/Linux
   chmod -R 755 backend/data
   chmod -R 755 backend/models
   ```

3. Verify the JSON data format by manually creating a test file.

## Model Retraining Issues

**Issue**: Model retraining doesn't work

**Solution**:

1. Make sure there are enough patient records for retraining.

2. Check the retraining threshold in `model_retrainer.py`.

3. Verify that the model files are being saved correctly.

4. Try manually triggering retraining:
   ```bash
   curl -X POST http://localhost:5000/api/retrain
   ```

## Still Having Issues?

If you're still experiencing problems after trying these solutions:

1. Check the application logs for more detailed error messages.

2. Try running the simplified test scripts to isolate the issue:

   ```bash
   cd backend
   python test_workflow.py
   ```

3. Make sure all system dependencies are installed (e.g., C++ compiler for some Python packages).

4. Consider creating a fresh virtual environment and reinstalling all dependencies.
