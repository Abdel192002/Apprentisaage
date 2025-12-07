@echo off
echo Starting MLflow UI (filesystem backend)...
echo URL: http://localhost:5000
echo Press Ctrl+C to stop
echo.

set PYTHONWARNINGS=ignore

mlflow ui --host 127.0.0.1 --port 5000

pause