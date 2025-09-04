#!/bin/bash

# Create necessary directories
mkdir -p backend/data/patients
mkdir -p backend/models

# Run the backend on port 8000
cd backend
python minimal_app_no_debug.py
