#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env. Put your GOOGLE_API_KEY in .env, then run again."
  exit 1
fi

if ! grep -Eq '^GOOGLE_API_KEY=.+$' .env || grep -Eq '^GOOGLE_API_KEY=(your_google_ai_api_key|YOUR_REAL_GOOGLE_AI_KEY)$' .env; then
  echo "Set GOOGLE_API_KEY in .env first."
  exit 1
fi

docker compose build
docker compose up -d

echo "Waiting for Langflow API..."
for _ in $(seq 1 60); do
  if curl -fsS http://localhost:7860/api/v1/auto_login >/dev/null 2>&1; then break; fi
  sleep 2
done

read -r -p "Load real Kaggle dataset now? (y/n; n uses sample fixture): " use_kaggle
if [[ "$use_kaggle" =~ ^[Yy]$ ]]; then
  docker compose exec langflow /app/.venv/bin/python /app/scripts/download_kaggle.py
  docker compose exec langflow /app/.venv/bin/python /app/scripts/prepare_db.py /app/data/kaggle_used_cars.csv --db /data/car_dealership.db
else
  docker compose exec langflow /app/.venv/bin/python /app/scripts/prepare_db.py /app/data/sample_used_cars.csv --db /data/car_dealership.db
fi

docker compose exec langflow /app/.venv/bin/python /app/scripts/build_and_install_flow.py

echo "DONE: Open http://localhost:7860"
echo "Generated flow file: flow/Car_Dealership_Agent.flow.json"
