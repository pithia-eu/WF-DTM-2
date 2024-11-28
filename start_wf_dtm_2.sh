#!/bin/bash

# Set the name of your venv
VENV_NAME=venv

# Navigate to project directory
cd /home/ubuntu/WF-DTM2

# Activate the venv
source $VENV_NAME/bin/activate

# Check if activation was successful
if [ $? -eq 0 ]
then
  echo "Virtual environment activated successfully."
else
  echo "Failed to activate virtual environment. Stop execution."
  exit
fi

# Run the uvicorn server
uvicorn main:app --reload --port 8085 --host 0.0.0.0