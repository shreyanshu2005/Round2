# BTIP Frontend — Backend Integration Guide

## What changed from the first draft

`app.js` now calls all six Layer 9 API endpoints at boot and on a 60-second refresh cycle.
Every value it can't load from the API falls back silently to the original mock values,
so the page always renders — even if the backend is down.

---

## Boot sequence

1. **`GET /auth/token`** — fetches a JWT using the demo Commander credentials.
   If the endpoint doesn't exist or returns an error the rest of the calls proceed
   without an `Authorization` header (useful while auth is still being wired up).

2. **`GET /api/v1/hotspots?limit=100`** — drives the City Pulse section:
   - City-wide risk index (average `risk_score` across clusters)
   - Active hotspot count + structural count (`persistence_score > 0.7`)
   - Top-pressure zone name
   - Network canvas node positions (mapped from `centroid_lat / centroid_lng`)
   - Intelligence card A: top cluster risk + zone name
   - Map focus card: top two cluster names as the "corridor"

3. **`GET /api/v1/risk?zone_id=1&shift=Evening&date=<today>`** — drives card B:
   - SHAP reasons list (top 3 `shap_explanations` items)
   - Confidence pill text

4. **`GET /api/v1/recommendations?shift=Evening&date=<today>&total_officers=20`** — drives the Action Plan section:
   - Orbit center officer/zone count
   - Orbit node labels (zone names + officer counts)
   - Recommendation card headings, officer counts, risk before/after, SHAP chips in accordion

5. **`POST /api/v1/simulation`** — fires when "Run Scenario" is clicked:
   - Reads `officerRange`, `zoneSelect`, and active shift button
   - Maps dropdown key → cluster_id by matching `zone_name` in the loaded hotspots
   - Animates `reductionValue`, `reliefValue`, `junctionValue`, `afterRisk`, P10/P50/P90

Polling: steps 2–4 repeat every 60 seconds automatically.

---

## API response shapes expected

### `/hotspots`
```json
[
  {
    "cluster_id": 7,
    "zone_name": "Silk Board Corridor",
    "centroid_lat": 12.9175,
    "centroid_lng": 77.6228,
    "violation_count": 312,
    "risk_score": 88,
    "persistence_score": 0.89
  }
]
```
The frontend also tolerates a wrapped shape `{ "hotspots": [...] }` or `{ "results": [...] }`.

### `/risk`
```json
{
  "zone_id": 1,
  "risk_score": 86,
  "confidence_band": { "p10": 12, "p50": 18.6, "p90": 24 },
  "predicted_violations": 143,
  "shap_explanations": [
    { "feature": "Rush hour overlap", "impact": 18, "direction": "+" },
    { "feature": "7-day repeat pattern", "impact": 14, "direction": "+" },
    { "feature": "Network centrality", "impact": 11, "direction": "+" }
  ]
}
```

### `/recommendations`
```json
[
  {
    "zone_id": 7,
    "zone_name": "Silk Board Corridor",
    "n_officers": 5,
    "risk_score": 88,
    "expected_reduction_pct": 31,
    "recommended_shift": "17:00-21:30",
    "shap_explanations": [...],
    "explanation": { "top_drivers": ["Friday evening pattern (+24%)", ...] }
  }
]
```

### `POST /simulation`
Request:
```json
{
  "zone_allocations": [{ "zone_id": 7, "n_officers": 20 }],
  "shift": "Evening",
  "date": "2025-01-15"
}
```
Response:
```json
{
  "total_reduction_pct": 23.4,
  "congestion_improvement_pct": 15.4,
  "affected_junction_count": 11,
  "confidence_band": { "p10": 17.8, "p50": 23.4, "p90": 29.0 }
}
```

---

## CORS

Add `localhost` (or your frontend origin) to the `allow_origins` list in `backend/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

The frontend runs on port **8080** (`python3 -m http.server 8080`).

---

## Auth credentials

The demo token call uses `username=commander&password=btip2025`.
Change these in `app.js` line ~20 to match whatever you hardcoded in `backend/core/config.py`.
If auth isn't ready yet, just comment out the `getToken` call — everything degrades gracefully.

---

## How to run

```bash
# Terminal 1 — backend
cd btip-gridlock2/backend
uvicorn main:app --reload --port 8000

# Terminal 2 — frontend
cd btip_frontend_integrated
python3 -m http.server 8080
# open http://localhost:8080
```

---

## What still uses mock values

| UI element | Status |
|---|---|
| Hero particle animation | Always procedural — no API needed |
| Traffic network canvas animation | Node positions are live from `/hotspots`; road/vehicle animation is always procedural |
| 24h forecast sparkline (card C) | Still SVG mock — wire to `/forecast` in the Next.js migration |
| Meter bar (`--value:88%`) | Still static — update after wiring `/violations` aggregate |
| "Rising 18% in 35 min" copy | Static marketing copy — update with real delta from two `/hotspots` calls |
