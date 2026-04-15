# Ensure execution policy allows script running
# Set-ExecutionPolicy Unrestricted -Scope CurrentUser

Write-Host "Activating Virtual Environment..." -ForegroundColor Green
$venvPath = Join-Path $PWD "venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    . $venvPath
} else {
    Write-Host "Warning: Virtual environment not found at $venvPath. Assuming global python." -ForegroundColor Yellow
}

Write-Host "Stopping any running background process on relevant ports..." -ForegroundColor Yellow
$ports = 8000, 8001, 8501, 8502, 8503
foreach ($p in $ports) {
    $pids = (Get-NetTCPConnection -LocalPort $p -ErrorAction SilentlyContinue).OwningProcess
    foreach ($id in $pids) {
        if ($id -and $id -ne 0) {
            Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
        }
    }
}

# The files now execute relative to the ROOt working directory due to centralized design!
$workingDir = $PWD.Path

Write-Host "Starting Travel ML Flask API..." -ForegroundColor Cyan
Start-Process -FilePath "python" -ArgumentList "api/flight_api.py" -WorkingDirectory $workingDir -WindowStyle Minimized

Write-Host "Starting Gender Classification Flask API..." -ForegroundColor Cyan
Start-Process -FilePath "python" -ArgumentList "api/gender_api.py" -WorkingDirectory $workingDir -WindowStyle Minimized

Start-Sleep -Seconds 2

Write-Host "Starting Travel ML Streamlit App on port 8501..." -ForegroundColor Magenta
Start-Process -FilePath "streamlit" -ArgumentList "run app/flight_ui.py --server.port=8501 --server.headless=true" -WorkingDirectory $workingDir -WindowStyle Minimized

Write-Host "Starting Gender Classification Streamlit App on port 8502..." -ForegroundColor Magenta
Start-Process -FilePath "streamlit" -ArgumentList "run app/gender_ui.py --server.port=8502 --server.headless=true" -WorkingDirectory $workingDir -WindowStyle Minimized

Write-Host "Starting Hotel Recommender Streamlit App on port 8503..." -ForegroundColor Magenta
Start-Process -FilePath "streamlit" -ArgumentList "run app/hotel_ui.py --server.port=8503 --server.headless=true" -WorkingDirectory $workingDir -WindowStyle Minimized

Write-Host "All 5 services have been restarted utilizing the new centralized MLOps directory structure." -ForegroundColor Green
Write-Host "You can close their respective windows to stop them." -ForegroundColor Yellow
