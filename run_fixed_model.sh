#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running Heart Failure Prediction System with Fixed Port${NC}"
echo "========================================================"

# Create necessary directories
echo -e "\n${YELLOW}Creating necessary directories...${NC}"
mkdir -p backend/data/patients
mkdir -p backend/models
echo -e "${GREEN}Directories created successfully!${NC}"

# Kill any existing processes on port 8080
echo -e "\n${YELLOW}Checking for existing processes on port 8080...${NC}"
lsof -ti:8080 | xargs kill -9 2>/dev/null
echo -e "${GREEN}Port 8080 is now available.${NC}"

# Start backend test server
echo -e "\n${YELLOW}Starting backend test server on port 8080...${NC}"
cd backend
python test_server.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo -e "${YELLOW}Waiting for backend to start...${NC}"
sleep 3

# Check if backend is running
if ps -p $BACKEND_PID > /dev/null; then
    echo -e "${GREEN}Backend test server started successfully!${NC}"
else
    echo -e "${RED}Failed to start backend test server.${NC}"
    exit 1
fi

# Start frontend development server
echo -e "\n${YELLOW}Starting frontend development server...${NC}"
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Function to handle script termination
function cleanup {
  echo -e "\n${YELLOW}Shutting down servers...${NC}"
  kill $BACKEND_PID 2>/dev/null
  kill $FRONTEND_PID 2>/dev/null
  echo -e "${GREEN}Servers shut down successfully!${NC}"
  exit
}

# Register the cleanup function for script termination
trap cleanup SIGINT SIGTERM

echo -e "\n${GREEN}Both servers are running!${NC}"
echo "Backend test server: http://localhost:8080"
echo "Frontend server: Check the URL in the npm output (typically http://localhost:5173)"
echo -e "${YELLOW}Press Ctrl+C to stop both servers.${NC}"
echo -e "\n${GREEN}Testing Instructions:${NC}"
echo "1. Enter patient data and submit the form to create new patient records"
echo "2. Go to the Model Training page to see the training history"
echo "3. Provide feedback on predictions to improve the model"
echo "4. Retrain the model to incorporate the feedback"
echo "5. Test the improved model with new patient data"
wait
