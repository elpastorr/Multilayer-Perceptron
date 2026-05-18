#!/bin/bash

if [ ! -d .venv ]; then
    echo "Creating virtual env..."
    python3 -m venv .venv
fi

.venv/bin/pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    .venv/bin/pip install -r requirements.txt
else
    echo "No requirements.txt found"
fi

echo "Requirements installed now activate venv with:"
echo "source .venv/bin/activate"
