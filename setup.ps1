Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Setting up the Football Prediction Project" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

Write-Host "`nChecking for existing virtual environment..."
if (-not (Test-Path "venv\Scripts\activate.ps1")) {
    Write-Host "Creating virtual environment venv without pip to avoid hangs..." -ForegroundColor Yellow
    python -m venv venv --without-pip
    
    Write-Host "Activating and installing pip manually..."
    . .\venv\Scripts\activate.ps1
    python -m ensurepip --upgrade
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor Green
    . .\venv\Scripts\activate.ps1
}

Write-Host "`nUpgrading pip to the latest version..."
python -m pip install --upgrade pip

Write-Host "`nInstalling dependencies..."
# Δοκιμάζει ΠΡΩΤΑ τον κεντρικό φάκελο και μετά τους υποφακέλους
 python -m pip install -r .\requirments\requirments.txt

Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "The virtual environment is active. To run the model, simply type:"
Write-Host "python main.py" -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Cyan