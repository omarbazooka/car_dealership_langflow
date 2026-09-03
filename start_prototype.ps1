$ErrorActionPreference = "Stop"

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env. Put your GOOGLE_API_KEY in .env, then run this script again." -ForegroundColor Yellow
    exit 1
}

$envText = Get-Content ".env" -Raw
if ($envText -match "GOOGLE_API_KEY=(your_google_ai_api_key|YOUR_REAL_GOOGLE_AI_KEY)?\s*(\r?\n|$)" -or $envText -notmatch "GOOGLE_API_KEY=.+") {
    Write-Host "Set GOOGLE_API_KEY in .env first." -ForegroundColor Yellow
    exit 1
}

docker compose build
docker compose up -d

Write-Host "Waiting for Langflow API..." -ForegroundColor Cyan
for ($i = 0; $i -lt 60; $i++) {
    try {
        Invoke-RestMethod -Uri "http://localhost:7860/api/v1/auto_login" -TimeoutSec 2 | Out-Null
        break
    } catch {
        Start-Sleep -Seconds 2
    }
}

$useKaggle = Read-Host "Load real Kaggle dataset now? (y/n; n uses sample fixture)"
if ($useKaggle -match "^[Yy]") {
    docker compose exec langflow /app/.venv/bin/python /app/scripts/download_kaggle.py
    docker compose exec langflow /app/.venv/bin/python /app/scripts/prepare_db.py /app/data/kaggle_used_cars.csv --db /data/car_dealership.db
} else {
    docker compose exec langflow /app/.venv/bin/python /app/scripts/prepare_db.py /app/data/sample_used_cars.csv --db /data/car_dealership.db
}

docker compose exec langflow /app/.venv/bin/python /app/scripts/build_and_install_flow.py

Write-Host "DONE: Open http://localhost:7860" -ForegroundColor Green
Write-Host "Generated flow file: flow/Car_Dealership_Agent.flow.json" -ForegroundColor Green
