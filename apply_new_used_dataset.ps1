$ErrorActionPreference = "Stop"

Write-Host "1/4 Building and validating Egypt cars dataset..." -ForegroundColor Cyan
docker compose exec -T langflow /app/.venv/bin/python /app/scripts/build_egypt_car_dataset.py --used /app/data/car_ads_details_kaggle.csv --out /app/data/egypt_cars_combined.csv --checkpoint /app/data/contactcars_new_checkpoint.jsonl --target-new 8374

Write-Host "2/4 Preparing SQLite database..." -ForegroundColor Cyan
docker compose exec -T langflow /app/.venv/bin/python /app/scripts/prepare_db.py /app/data/egypt_cars_combined.csv --db /data/car_dealership.db

Write-Host "3/4 Building and installing Langflow flow..." -ForegroundColor Cyan
docker compose exec -T langflow /app/.venv/bin/python /app/scripts/build_and_install_flow.py

Write-Host "4/4 Running Smoke & E2E Validation..." -ForegroundColor Cyan
docker compose exec -T langflow /app/.venv/bin/python /app/scripts/smoke_test.py
docker compose exec -T langflow /app/.venv/bin/python /app/scripts/e2e_validate.py

Write-Host "Done!" -ForegroundColor Green
