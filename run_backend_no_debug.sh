#!/bin/bash

# Create necessary directories
mkdir -p backend/data/patients
mkdir -p backend/models

# Run the backend without debug mode
cd backend
python minimal_app_no_debug.py
