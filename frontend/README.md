# ForecastPH frontend

This is the Next.js 14 App Router website for ForecastPH. It reads generated operational JSON from `public/forecasts/`; model training and inference run in the repository's backend automation, not in Vercel or the browser.

## Local development

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

## Routes

- `/` — market overview
- `/companies` — tracked company directory
- `/companies/<SYMBOL>` — company forecast, metrics, and charts
- `/watchlist` — browser-local watchlist
- `/compare` — current operational Models dashboard
- `/learn-stocks` — educational stock and forecast guide
- `/learn` — redirect to `/learn-stocks`
- `/about` — project architecture and limitations

The AI assistant receives page-aware context built from the same operational forecast documents.

## Forecast data

```text
public/forecasts/
├── companies.json
├── dashboard.json
├── latest.json
├── metrics.json
├── company/<SYMBOL>.json
└── history/<SYMBOL>.json
```

The backend exporter generates and validates this complete tree. See `../docs/frontend-forecast-contract.md` for schemas and cross-file invariants.

The Models page uses current per-company evaluation metrics to derive model comparisons and RMSE win counts. It does not require a separate fixed-study JSON document.

## Production validation

```bash
npm run build
npx tsc --noEmit
```

## Vercel

Set the Vercel project Root Directory to `frontend/`. Forecast JSON is committed by the GitHub Actions pipelines before Vercel builds the site. The application does not need a forecasting API or Python runtime.

The AI route requires its configured provider credentials in the deployment environment. Forecast pages themselves read only repository JSON.
