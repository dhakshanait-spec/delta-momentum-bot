# Delta Momentum Bot (Railway-ready)

1. Create a GitHub repo and push the files: `main.py`, `requirements.txt`, `Procfile`, `.env` (from `.env.example`).
2. Sign into Railway (https://railway.app) and create a new project from GitHub — select this repo.
3. Set environment variables in Railway (in Settings → Variables) using the values from `.env`.
4. Deploy. Railway will install requirements and start the `worker` defined in `Procfile`.
5. Keep PAPER_MODE=true for testing. When ready, set PAPER_MODE=false.

Notes:
- Test thoroughly on paper or sandbox keys before live trading.
- Tune MIN_AVG_VOLUME, VOLUME_MULTIPLIER, VWAP_WINDOW and risk sizing.
