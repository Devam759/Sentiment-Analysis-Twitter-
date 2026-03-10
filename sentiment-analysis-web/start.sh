#!/bin/bash

echo "Starting Twitter Sentiment Analysis Web Application..."
echo ""

echo "Step 1: Starting Flask Backend Server..."
cd backend && python app.py &
BACKEND_PID=$!

echo "Step 2: Starting React Frontend Server..."
sleep 3
cd .. && npm run dev &
FRONTEND_PID=$!

echo ""
echo "Both servers are starting..."
echo "Frontend: http://localhost:5173"
echo "Backend: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for user to stop
wait $BACKEND_PID $FRONTEND_PID
