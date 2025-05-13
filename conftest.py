"""Configuration file for pytest."""

import sys
import os

# Add the parent directory to sys.path to allow imports from the src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 