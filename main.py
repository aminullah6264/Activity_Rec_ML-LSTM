#!/usr/bin/env python3
"""
Main entry point for the Activity Recognition system.

This script provides a simple interface to the modernized activity recognition
system using temporal optical flow features and multi-layer LSTM.
"""

import sys
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from activity_recognition.cli import cli

if __name__ == "__main__":
    cli()