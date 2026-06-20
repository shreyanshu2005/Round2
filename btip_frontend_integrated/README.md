# BTIP Frontend — First Visual Draft

A zero-dependency interactive frontend concept for the Bengaluru Traffic Intelligence Platform.

## Why this stack

This first draft uses plain HTML, CSS and JavaScript so it can be opened immediately, reviewed quickly, and later migrated into React/Vite, Next.js, SvelteKit or another production framework after the backend contracts are available.

## Run locally

Option 1: open `index.html` directly in a modern browser.

Option 2 (recommended):

```bash
cd btip_frontend_draft
python3 -m http.server 8080
```

Then open `http://localhost:8080`.

## Included interactions

- Procedural animated Bengaluru-style network visualization
- Switchable risk / violation / traffic-flow layers
- Motion toggle and reduced-motion support
- Animated metrics and live clock
- Digital-twin scenario controls
- Draggable before/after comparison map
- Officer allocation orbit
- Expandable recommendation explanations
- Responsive desktop, tablet and mobile layouts

## Backend integration points

The prototype currently uses representative mock values. Replace those with API calls when the FastAPI backend is available:

- `GET /api/v1/violations`
- `GET /api/v1/hotspots`
- `GET /api/v1/risk`
- `GET /api/v1/forecast`
- `GET /api/v1/recommendations`
- `POST /api/v1/simulation`

A production migration should split the page into reusable components and add typed API models, loading states, empty states, error handling and authentication.
