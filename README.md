# FanCart

FanCart is a sports-merchandise demand forecaster. A Flask application serves a responsive HTML/CSS/vanilla JavaScript dashboard and a JSON API. The existing Python data-generation, feature-engineering, model-training and forecasting logic remains in place.

## Setup

Python 3.10 or later is recommended. The current development environment uses Python 3.13.14. From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). FanCart loads the saved model from `models/` and its validation metrics from `metrics.json`. If the model artifact is missing, training runs in a background thread; model-dependent API requests return `503` with a training status message until it is ready. The app uses CPU-only scikit-learn models. Network access is optional: when Cricsheet cannot be reached or parsed, generated sample match data is used.

## Deploying on Render

Create a Render Web Service for this repository and configure:

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `gunicorn app:app`
- **Environment:** set `SECRET_KEY` to a long, random value. Render's `RENDER` environment variable enables `Secure`, `HttpOnly`, `SameSite=Lax` session and remember-me cookies.

The trained `models/fancart_model.joblib` bundle and `metrics.json` are part of the project and are not ignored by `.gitignore`. Keep both artifacts in the deployed source. `GET /healthz` is an unauthenticated health check. For local `python app.py`, the server binds to `127.0.0.1:5000`; when `PORT` is set, it binds to `0.0.0.0` and uses that port.

## Architecture

- `app.py` provides the Flask static front end, authentication endpoints and authenticated forecasting API.
- `static/` contains the single-page HTML/CSS/vanilla JavaScript client. Charts use the locally bundled Chart.js library; a canvas renderer is available as a fallback.
- `auth/db.py` and `auth/service.py` maintain the SQLite users and remember-token tables. Passwords are PBKDF2-HMAC-SHA256 hashes with random salts. The browser receives a signed Flask session cookie and, when requested, a separate HttpOnly remember-me token; only its SHA-256 digest is stored in SQLite.
- `data/loader.py` fetches Cricsheet IPL JSON and falls back to generated results. `data/synthetic.py` provides sample match schedules and synthetic product sales.
- `scripts/export_sample_csv.py` exports `sample_sales_data.csv` using the active match loader and the same synthetic sales effects. Run `python scripts/export_sample_csv.py` to regenerate the downloadable sample.
- `ml/features.py`, `ml/train.py` and `ml/predict.py` build date-safe features, train median and 10th/90th percentile regressors, and produce forecasts and stock recommendations.
- `models/` stores the model bundle; `metrics.json` stores its chronological validation metrics.

## API

All routes return JSON except `/`, which serves the application. Authentication routes are public; every other `/api` route requires a session or valid remember-me cookie.

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/auth/signup` | Create an account and start a session. JSON: `username`, `password`. |
| `POST` | `/api/auth/login` | Sign in. JSON: `username`, `password`, optional boolean `remember`. |
| `POST` | `/api/auth/demo` | Start a session for the demo user. |
| `POST` | `/api/auth/logout` | Revoke the remember-me token and clear cookies. |
| `GET` | `/api/auth/me` | Return the current user, or `401`. |
| `GET` | `/api/data/sample` | Download the reproducible, match-linked sample sales CSV. |
| `GET` | `/api/meta` | Teams, products, next-30-day fixtures, match-data source and synthetic-sales note. |
| `GET` | `/api/kpis` | Fourteen-day forecast units, stockout-risk count and next-match uplift. |
| `GET` | `/api/forecast?team=&product=&days=` | Last 90 days of actual sales and forecast with 10th/90th percentile bounds. |
| `POST` | `/api/whatif` | Win/loss demand scenario, uplift duration and stock recommendation. JSON: `team`, `product`, `match_date`, `outcome`, `lead_time`. |
| `GET` | `/api/report` | Saved model/baseline validation metrics and permutation feature importance. |
| `GET` | `/api/insights` | Synthetic-sales comparisons for match results and match days. |

Invalid request values return a JSON `error` and an appropriate 4xx status. An unauthenticated protected request returns `401`.

## Data sources and licensing

Match results are downloaded from [Cricsheet](https://cricsheet.org/) when available. Cricsheet data is provided under the [Open Data Commons Attribution License (ODC-By)](https://opendatacommons.org/licenses/by/1-0/). The app shows whether matches came from Cricsheet or generated sample data. Upcoming fixtures are simulated, not an official schedule.

## Limitations

All merchandise sales and prices are synthetic; the project has no real store, transaction, inventory or player-performance data. The stockout KPI uses recent synthetic sales as a proxy because actual on-hand inventory is unavailable. The forecasts and scenario effects are illustrative, not business guarantees. Match-day and post-result effects are patterns embedded in the synthetic data.

## Five-minute demo

1. Run `python app.py` and open the local address above.
2. Select **Continue as Demo User**.
3. On **Forecast**, choose a team and product; inspect actual history, forecast and uncertainty band.
4. On **What-If**, choose an upcoming match, select Win or Loss, set the stock lead time and calculate the modelled demand and stock recommendation.
5. On **Model Report**, compare validation metrics with the seasonal-naive baseline and inspect permutation importance. Finish on **Insights** to see descriptive win/loss and match-day comparisons, noting that sales data is synthetic.
