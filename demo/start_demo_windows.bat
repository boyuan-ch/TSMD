@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel% equ 0 (
    set "PYTHON_LAUNCHER=py"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Python 3 was not found.
        echo Install Python from https://www.python.org/downloads/windows/
        echo During installation, enable "Add python.exe to PATH".
        pause
        exit /b 1
    )
    set "PYTHON_LAUNCHER=python"
)

echo Starting the IR9P demo server...
if "%PYTHON_LAUNCHER%"=="py" (
    start "IR9P Demo Server" /min cmd /k py -3 serve.py --host 127.0.0.1 --port 8765
) else (
    start "IR9P Demo Server" /min cmd /k python serve.py --host 127.0.0.1 --port 8765
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$deadline = (Get-Date).AddSeconds(15);" ^
  "do {" ^
  "  try {" ^
  "    $response = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8765/' -TimeoutSec 1;" ^
  "    if ($response.StatusCode -eq 200) { exit 0 }" ^
  "  } catch {}" ^
  "  Start-Sleep -Milliseconds 250" ^
  "} while ((Get-Date) -lt $deadline);" ^
  "exit 1"

if errorlevel 1 (
    echo.
    echo The local demo server did not start on port 8765.
    echo Check the "IR9P Demo Server" window for the Python error.
    pause
    exit /b 1
)

echo Demo server is ready. Opening the browser...
start "" "http://127.0.0.1:8765/"
exit /b 0
