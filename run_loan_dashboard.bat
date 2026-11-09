@echo off
cd /d "%~dp0"
echo Starting Personal Loan dashboard...
echo Open http://localhost:8502 in your browser (first load takes ~5 seconds).
start "" http://localhost:8502
".venv\Scripts\streamlit.exe" run personal_loan/app.py --server.port 8502
pause
