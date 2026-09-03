# Egypt Cars Dataset + Web Fallback Patch

This patch upgrades the existing Langflow car-dealership prototype in two ways:

1. **Balanced Egyptian inventory dataset**
   - Used source: the supplied Kaggle `car_ads_details_kaggle.csv`.
   - Exact supplied used row count: **8,374**.
   - New source: **public ContactCars brand-new dealer listing pages**.
   - Target: **8,374 unique NEW rows**, no artificial duplication.
   - Final target: **16,748 rows** plus the CSV header.
   - Adds `Condition` with `used` / `new`.

2. **Web fallback for facts outside the DB**
   - The Langflow flow adds built-in `Web Search` + `URL` tools.
   - If a car is already identified from the DB but a requested fact is absent (e.g. acceleration, dimensions, airbags, ADAS, warranty, boot size), the agent must search the web instead of guessing.
   - Prompt source priority: ContactCars listing/model/trim page → official Egypt manufacturer/importer → reputable automotive source.

## Combined CSV schema

The original 9 Kaggle columns remain first:

- Brand
- Model
- Kilometers
- Year
- Fuel Type
- Transmission Type
- Engine Capacity (CC)
- Body Type
- Price_EGP

Added columns:

- Condition
- Trim
- Location
- Origin
- Horsepower
- Color
- Source
- Source_URL
- Source_ID
- Collected_At

## Important data-quality note

The source CSV has 8,374 rows. 58 rows are missing Brand and/or Model. They are preserved in the final CSV so the requested used/new balance remains exact. The SQLite importer intentionally skips rows that cannot be identified by both brand and model.

## How to apply

Extract/copy this patch **over the existing project root**, preserving folders. It is designed as an overlay and does not require a Docker image rebuild.

From PowerShell in the folder containing `docker-compose.yml`:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\apply_new_used_dataset.ps1
```

Manual equivalent:

```powershell
docker compose exec langflow /app/.venv/bin/python /app/scripts/build_egypt_car_dataset.py --used /app/data/car_ads_details_kaggle.csv --out /app/data/egypt_cars_combined.csv --checkpoint /app/data/contactcars_new_checkpoint.jsonl --target-new 8374

docker compose exec langflow /app/.venv/bin/python /app/scripts/prepare_db.py /app/data/egypt_cars_combined.csv --db /data/car_dealership.db

docker compose restart langflow

docker compose exec langflow /app/.venv/bin/python /app/scripts/build_and_install_flow.py
```

## Collector behavior

- Reads ContactCars `robots.txt` before collecting.
- Uses public pages only.
- Does not bypass login, CAPTCHA, rate limits, or access controls.
- Discovers listing URLs through public sitemap/catalogue/model pages.
- Uses a checkpoint: `data/contactcars_new_checkpoint.jsonl`.
- If interrupted, rerun the same command and it resumes.
- Every accepted NEW row must have a unique ContactCars `Source_URL`/`Source_ID`.
- If fewer than 8,374 usable real new listings are publicly discoverable, the script exits without fabricating/duplicating rows and writes `data/egypt_cars_combined.report.json`.

## Expected flow changes

The generated flow contains:

- Chat Input
- Input Guardrails
- Gemini Car Sales Agent + Memory
- Search Cars (New + Used)
- Get Car Details
- Compare Cars
- Dealership Knowledge RAG
- Create Test Drive
- Create Sales Lead
- Web Search
- URL
- Output Guardrails
- Chat Output

The class name `SearchUsedCars` is intentionally kept internally for compatibility with the existing builder, while its UI display name is now **Search Cars (New + Used)**.
