#!/bin/bash

# Restart Redis server
brew services restart redis

# Restart PostgreSQL server
brew services restart postgresql

# Run the main program
python main.py