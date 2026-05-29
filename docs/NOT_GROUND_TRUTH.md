# ⚠️ AI-generated study material — not a source of truth

This folder is the rendered output of an **AI-built study companion** for GRA 4160 (Machine Learning, BI Norwegian Business School). Treat it as a **study companion, not a source of truth**.

Every page is *derived* from course material and textbooks — it is not itself authoritative. Verify claims (formulas, code, hyperparameters, dataset details) against the underlying source material using the citations on each page. **That source material is NOT in this repo** — the lecture slides, notebooks, exercises, past exams, and the ESL/ISL textbooks must be obtained separately.

## Known limitations

- `methods/nn_pytorch.html` §4 has qualitative numbers only (PyTorch wasn't installed when the site was built).
- Two datasets (`SMSSpamCollection.csv`, `bank-additional-full.csv`) weren't available at build time; those drill pages show idioms without computed outputs.
- The site uses Observable JS and **requires an HTTP server** — opening these HTML files directly via `file://` will not work.

## Viewing locally

The site must be **served, not opened directly**. From this `_site/` folder:

```powershell
python -m http.server 8000
```

Then open **http://localhost:8000/**. Stop with `Ctrl+C`.
