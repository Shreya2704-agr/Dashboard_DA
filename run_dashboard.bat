@echo off
cd /d "%~dp0"
echo Starting Campaign Intelligence dashboard...
echo Open http://localhost:8501 in your browser (first load takes ~20 seconds).
start "" http://localhost:8501
".venv\Scripts\streamlit.exe" run app.py --server.port 8501
pause
