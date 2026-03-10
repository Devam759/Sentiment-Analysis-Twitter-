@echo off
echo Starting Twitter Sentiment Analysis Web Application...
echo.

echo Step 1: Starting Flask Backend Server...
start cmd /k "cd backend && python app.py"

echo Step 2: Starting React Frontend Server...
timeout /t 3 /nobreak >nul
start cmd /k "npm run dev"

echo.
echo Both servers are starting...
echo Frontend: http://localhost:5173
echo Backend: http://localhost:5000
echo.
echo Press any key to exit...
pause >nul
