#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Heart Failure Prediction System Setup${NC}"
echo "========================================"

# Create necessary directories
echo -e "\n${YELLOW}Creating necessary directories...${NC}"
mkdir -p backend/data/patients
mkdir -p backend/models
echo -e "${GREEN}Directories created successfully!${NC}"

# Check Python version
echo -e "\n${YELLOW}Checking Python version...${NC}"
python --version
if [ $? -ne 0 ]; then
    echo -e "${RED}Python not found! Please install Python 3.8 or higher.${NC}"
    exit 1
fi

# Check Node.js version
echo -e "\n${YELLOW}Checking Node.js version...${NC}"
node --version
if [ $? -ne 0 ]; then
    echo -e "${RED}Node.js not found! Please install Node.js 14 or higher.${NC}"
    exit 1
fi

# Install backend dependencies
echo -e "\n${YELLOW}Installing backend dependencies...${NC}"
cd backend
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install backend dependencies!${NC}"
    exit 1
fi
echo -e "${GREEN}Backend dependencies installed successfully!${NC}"
cd ..

# Install frontend dependencies
echo -e "\n${YELLOW}Installing frontend dependencies...${NC}"
cd frontend
npm install
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install frontend dependencies!${NC}"
    exit 1
fi
echo -e "${GREEN}Frontend dependencies installed successfully!${NC}"
cd ..

# Verify backend installation
echo -e "\n${YELLOW}Verifying backend installation...${NC}"
cd backend
python verify_installation.py
cd ..

# Start backend server
echo -e "\n${YELLOW}Starting backend server...${NC}"
cd backend
python app.py &
BACKEND_PID=$!
cd ..

# Start frontend development server
echo -e "\n${YELLOW}Starting frontend development server...${NC}"
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Function to handle script termination
function cleanup {
  echo -e "\n${YELLOW}Shutting down servers...${NC}"
  kill $BACKEND_PID
  kill $FRONTEND_PID
  echo -e "${GREEN}Servers shut down successfully!${NC}"
  exit
}

# Register the cleanup function for script termination
trap cleanup SIGINT SIGTERM

echo -e "\n${GREEN}Both servers are running!${NC}"
echo "Backend server: http://localhost:5000"
echo "Frontend server: Check the URL in the npm output (typically http://localhost:5173)"
echo -e "${YELLOW}Press Ctrl+C to stop both servers.${NC}"
wait
