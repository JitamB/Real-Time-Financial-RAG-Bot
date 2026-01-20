#!/bin/bash

# Start the Backend (Pathway) in the background
echo "Starting Backend (app.py)..."
python app.py &

# Start the Frontend (Streamlit) in the foreground
echo "Starting Dashboard..."
python -m streamlit run dashboard.py --server.port=$PORT --server.address=0.0.0.0
