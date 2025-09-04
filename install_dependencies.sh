#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Step-by-Step Package Installation${NC}"
echo "========================================"

# Function to install a package with error handling
install_package() {
    package=$1
    echo -e "\n${YELLOW}Installing $package...${NC}"
    pip install $package
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Successfully installed $package!${NC}"
        return 0
    else
        echo -e "${RED}Failed to install $package.${NC}"
        echo -e "${YELLOW}Trying alternative installation methods...${NC}"
        
        # Try with --no-deps flag
        echo "Trying with --no-deps flag..."
        pip install --no-deps $package
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Successfully installed $package with --no-deps!${NC}"
            return 0
        else
            echo -e "${RED}Failed to install $package with --no-deps.${NC}"
            return 1
        fi
    fi
}

# Essential packages
essential_packages=("flask" "flask-cors" "numpy" "pandas" "scikit-learn" "joblib")

# Optional packages
optional_packages=("xgboost" "shap" "neurokit2" "matplotlib")

# Install essential packages
echo -e "\n${YELLOW}Installing essential packages...${NC}"
for package in "${essential_packages[@]}"; do
    install_package $package
done

# Ask if user wants to install optional packages
echo -e "\n${YELLOW}Do you want to install optional packages? (y/n)${NC}"
read -p "These include xgboost, shap, neurokit2, and matplotlib: " answer

if [[ $answer == "y" || $answer == "Y" ]]; then
    echo -e "\n${YELLOW}Installing optional packages...${NC}"
    for package in "${optional_packages[@]}"; do
        install_package $package
    done
else
    echo -e "\n${YELLOW}Skipping optional packages.${NC}"
    echo "Note: Some functionality may be limited without these packages."
fi

echo -e "\n${GREEN}Installation process completed!${NC}"
echo "Check the output above for any packages that failed to install."
echo "You can try installing those packages manually if needed."
