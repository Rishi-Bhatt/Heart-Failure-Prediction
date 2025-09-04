#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running Updated Heart Failure Prediction System${NC}"
echo "========================================================"

# Create necessary directories
echo -e "\n${YELLOW}Creating necessary directories...${NC}"
mkdir -p backend/data/patients
mkdir -p backend/models
echo -e "${GREEN}Directories created successfully!${NC}"

# Start backend test server
echo -e "\n${YELLOW}Starting updated backend test server...${NC}"
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
  kill $BACKEND_PID
  kill $FRONTEND_PID
  echo -e "${GREEN}Servers shut down successfully!${NC}"
  exit
}

# Register the cleanup function for script termination
trap cleanup SIGINT SIGTERM

echo -e "\n${GREEN}Both servers are running!${NC}"
echo "Backend test server: http://localhost:8000"
echo "Frontend server: Check the URL in the npm output (typically http://localhost:5173)"
echo -e "${YELLOW}Press Ctrl+C to stop both servers.${NC}"
wait
