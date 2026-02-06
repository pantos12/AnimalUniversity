#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"

echo "Creating venv (.venv)..."
$PYTHON -m venv .venv

echo "Upgrading pip..."
.venv/bin/python -m pip install -U pip

if [ -f "requirements.txt" ]; then
  echo "Installing requirements..."
  .venv/bin/python -m pip install -r requirements.txt
else
  echo "requirements.txt not found. Nothing to install."
fi

echo "Python version:"
.venv/bin/python --version

echo "Installed packages (top-level):"
.venv/bin/python -m pip list
