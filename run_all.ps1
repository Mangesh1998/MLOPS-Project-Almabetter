$baseDir = $PSScriptRoot
$venvPython = "$baseDir\venv\Scripts\python.exe"
$venvStreamlit = "$baseDir\venv\Scripts\streamlit.exe"

Write-Host "Starting Travel ML Flask API..."
Start-Process -FilePath $venvPython -ArgumentList "app.py" -WorkingDirectory "$baseDir\Travel_ML_System" -WindowStyle Normal

Write-Host "Starting Gender Classification Flask API..."
Start-Process -FilePath $venvPython -ArgumentList "app.py" -WorkingDirectory "$baseDir\Gender Classification Model" -WindowStyle Normal

Write-Host "Starting Travel ML Streamlit App on port 8501..."
Start-Process -FilePath $venvStreamlit -ArgumentList "run streamlit_app.py --server.port 8501" -WorkingDirectory "$baseDir\Travel_ML_System" -WindowStyle Normal

Write-Host "Starting Gender Classification Streamlit App on port 8502..."
Start-Process -FilePath $venvStreamlit -ArgumentList "run streamlit_app.py --server.port 8502" -WorkingDirectory "$baseDir\Gender Classification Model" -WindowStyle Normal

Write-Host "Starting Hotel Recommender Streamlit App on port 8503..."
Start-Process -FilePath $venvStreamlit -ArgumentList "run streamlit_app.py --server.port 8503" -WorkingDirectory "$baseDir\Hotel Recommender System" -WindowStyle Normal

Write-Host "All 5 services have been restarted with corrected working directories."
Write-Host "You can close their respective windows to stop them."
