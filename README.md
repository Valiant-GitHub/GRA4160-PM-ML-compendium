# GRA 4160 — Interactive Study Companion

Built by **Vilijam Cekov** as a personal study aid for GRA 4160 (Machine Learning) at BI Norwegian Business School.

This is an AI-assisted study companion built from the course material (lecture slides, notebooks, past exams) and the two recommended textbooks (ESL and ISL). See `CLAUDE.md` and `NOT_GROUND_TRUTH.md` for build notes and source-fidelity rules.

## Quick start

To view the rendered site locally:

```powershell
cd _site
python -m http.server 8000
```

Then open `http://localhost:8000/`.

To re-render the site (requires Quarto + the .venv from `requirements-build.txt`):

```powershell
$env:QUARTO_PYTHON = ".venv\Scripts\python.exe"
quarto render
```
