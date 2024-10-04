#!/bin/bash

# Restart Redis server
brew services restart redis || { echo "Error restarting Redis. Exiting."; exit 1; }

# Restart PostgreSQL server
brew services restart postgresql || { echo "Error restarting PostgreSQL. Exiting."; exit 1; }

# Run the main program
python main.py