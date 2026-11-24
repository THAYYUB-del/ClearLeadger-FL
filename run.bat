@echo off
setlocal

echo ========================================================
echo   ClearLedger-FL — Local Windows Launcher (CPU Only)
echo ========================================================
echo.

if not exist ".env" (
  echo [INFO] Creating .env from .env.example...
  copy ".env.example" ".env" >nul
)

echo [1/2] Starting Python FastAPI Backend on http://127.0.0.1:8000 ...
start "ClearLedger-FL Backend" cmd /k "cd backend && if not exist .venv (python -m venv .venv) && call .venv\Scripts\activate && pip install -r requirements.txt && uvicorn main:app --host 127.0.0.1 --port 8000"

timeout /t 4 /nobreak >nul

echo [2/2] Starting React + Vite Frontend on http://localhost:5173 ...
set VITE_API_PROXY=http://127.0.0.1:8000
start "ClearLedger-FL Frontend" cmd /k "npm install && npm run dev:frontend"

echo.
echo ClearLedger-FL services launched.
echo Frontend UI: http://localhost:5173
echo FastAPI Docs: http://127.0.0.1:8000/docs
endlocal
