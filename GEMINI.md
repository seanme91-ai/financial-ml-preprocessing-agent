# Project Rules: Auto-Sync to GitHub

Whenever you make verified edits to files in this project (`financial-ml-preprocessing-agent`):

1. **Verify before pushing**:
   - After editing `pipeline.py`, run `py pipeline.py --skip-fetch --skip-upload` and confirm it exits with code 0.
   - After editing `fetch_daily_data.py`, run `py fetch_daily_data.py --dry-run` and confirm it exits with code 0.
2. **Never commit secrets or generated artifacts**: Ensure `.env`, `*.json` credential files, and `data/processed/` remain untracked/ignored.
3. **Auto-commit and push to GitHub**: Unless the user explicitly asks not to push, automatically stage the changed files, create a clear commit message, and push to `origin main`:
   ```powershell
   git status
   git add -A
   git commit -m "<concise description of changes>"
   git push origin main
   ```
   Pushing to `main` automatically triggers the GitHub Actions workflow (`.github/workflows/run-pipeline.yml`).
4. **Report sync status**: Briefly confirm the commit hash and that the update was pushed to GitHub.
