#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up Conda Environment for Heart Failure Prediction System${NC}"
echo "========================================================"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Conda is not installed. Please install Miniconda or Anaconda first.${NC}"
    echo "You can download Miniconda from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
echo -e "\n${YELLOW}Creating conda environment 'heart_failure'...${NC}"
conda create -y -n heart_failure python=3.8
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to create conda environment.${NC}"
    exit 1
fi
echo -e "${GREEN}Conda environment created successfully!${NC}"

# Activate conda environment
echo -e "\n${YELLOW}Activating conda environment...${NC}"
eval "$(conda shell.bash hook)"
conda activate heart_failure
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to activate conda environment.${NC}"
    exit 1
fi
echo -e "${GREEN}Conda environment activated!${NC}"

# Install essential packages
echo -e "\n${YELLOW}Installing essential packages...${NC}"
conda install -y flask flask-cors numpy pandas scikit-learn joblib
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install essential packages.${NC}"
    echo -e "${YELLOW}Trying to install with pip...${NC}"
    pip install flask flask-cors numpy pandas scikit-learn joblib
fi

# Ask if user wants to install optional packages
echo -e "\n${YELLOW}Do you want to install optional packages? (y/n)${NC}"
read -p "These include xgboost, shap, neurokit2, and matplotlib: " answer

if [[ $answer == "y" || $answer == "Y" ]]; then
    echo -e "\n${YELLOW}Installing optional packages...${NC}"
    
    # Try to install with conda first
    conda install -y xgboost matplotlib
    
    # Try to install remaining packages with pip
    pip install shap neurokit2
else
    echo -e "\n${YELLOW}Skipping optional packages.${NC}"
    echo "Note: Some functionality may be limited without these packages."
fi

# Create directories
echo -e "\n${YELLOW}Creating necessary directories...${NC}"
mkdir -p backend/data/patients
mkdir -p backend/models
echo -e "${GREEN}Directories created successfully!${NC}"

echo -e "\n${GREEN}Setup completed!${NC}"
echo -e "To activate this environment in the future, run: ${YELLOW}conda activate heart_failure${NC}"
echo -e "To run the minimal backend, run: ${YELLOW}cd backend && python minimal_app.py${NC}"
echo -e "To run the frontend, run: ${YELLOW}cd frontend && npm run dev${NC}"
