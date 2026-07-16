@echo off
setlocal
REM Drag-and-drop a folder onto this .bat, or run: extract_prompts.bat "C:\path\to\images"
REM Writes prompts.txt and prompts.csv into the target folder.

if "%~1"=="" (
    echo Usage: %~nx0 "C:\path\to\images"
    echo        ^(or drag-and-drop a folder onto this file^)
    pause
    exit /b 1
)

set "TARGET=%~1"

if not exist "%TARGET%\" (
    echo ERROR: Folder does not exist: %TARGET%
    pause
    exit /b 1
)

set "SCRIPT_DIR=%~dp0"

echo Scanning "%TARGET%" ...
python "%SCRIPT_DIR%extract_prompts.py" "%TARGET%" "%TARGET%\prompts.txt"
python "%SCRIPT_DIR%extract_prompts.py" "%TARGET%" "%TARGET%\prompts.csv" --csv

echo.
echo Done. prompts.txt and prompts.csv written to "%TARGET%"
pause
endlocal
