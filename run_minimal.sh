#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running Minimal Version of Heart Failure Prediction System${NC}"
echo "========================================================"

# Create necessary directories
echo -e "\n${YELLOW}Creating necessary directories...${NC}"
mkdir -p backend/data/patients
mkdir -p backend/models
echo -e "${GREEN}Directories created successfully!${NC}"

# Check if Flask is installed
echo -e "\n${YELLOW}Checking if Flask is installed...${NC}"
python -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}Flask is not installed. Installing minimal requirements...${NC}"
    pip install flask flask-cors
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install Flask. Please install it manually:${NC}"
        echo "pip install flask flask-cors"
        exit 1
    fi
fi
echo -e "${GREEN}Flask is installed!${NC}"

# Start backend server
echo -e "\n${YELLOW}Starting minimal backend server...${NC}"
cd backend
python minimal_app.py &
BACKEND_PID=$!
cd ..

# Check if Node.js is installed
echo -e "\n${YELLOW}Checking if Node.js is installed...${NC}"
node --version >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo -e "${RED}Node.js is not installed. Please install it to run the frontend.${NC}"
    echo "You can download Node.js from: https://nodejs.org/"
    echo -e "${YELLOW}The backend server is still running at http://localhost:5000${NC}"
    echo -e "Press Ctrl+C to stop the backend server."
    wait $BACKEND_PID
    exit 1
fi
echo -e "${GREEN}Node.js is installed!${NC}"

# Check if frontend dependencies are installed
echo -e "\n${YELLOW}Checking if frontend dependencies are installed...${NC}"
if [ ! -d "frontend/node_modules" ]; then
    echo -e "${YELLOW}Frontend dependencies not found. Installing...${NC}"
    cd frontend
    npm install
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install frontend dependencies.${NC}"
        echo -e "${YELLOW}The backend server is still running at http://localhost:5000${NC}"
        echo -e "Press Ctrl+C to stop the backend server."
        cd ..
        wait $BACKEND_PID
        exit 1
    fi
    cd ..
fi
echo -e "${GREEN}Frontend dependencies are installed!${NC}"

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
