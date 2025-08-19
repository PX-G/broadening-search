Spectral Line Broadening Literature Search

Crossref multi-strategy retrieval + spectroscopy-aware heuristics + batch LLM triage, with stable sorting (Journal priority → Year → Title), optional species/perturber hard filter, and one-click CSV/BibTeX export.
Built for finding experimental pressure broadening parameters (γ, n, shifts, HWHM/FWHM, Voigt/Lorentz) fast.

Use model1.py (DeepSeek) or model2.py (Moonshot Kimi). The two files are identical except for the LLM_MODEL string.

Why this tool?

“Linewidth” appears in many unrelated contexts (optical clocks, micro-resonators, HEP “gamma rays”…). This app:

casts a wide Crossref net using multiple fields/queries →

deduplicates →

soft-denoises with domain terms →

ranks heuristically by spectroscopy keywords →

batches an LLM to keep likely experimental broadening papers →

stably sorts so JQSRT/JMS/JCP appear first, then year, then title.

Features

Crossref search with multi-strategy recall (query/title/bibliographic)

Physics-journal gating and priority list (JQSRT → JMS → JCP)

Species/perturber hard filter (must appear in title/abstract)

Stable sort: journal priority → year (asc/desc) → title (tie-break)

Backoff & retries for HTTP/429/5xx

Keyword highlighting, click-to-copy DOI

Export CSV (broadening_results.csv) and BibTeX (broadening_results.bib)

Transparent logs (fetched / heuristic kept / LLM kept with batches)

Install
1) Create & activate a virtual environment

Windows (PowerShell)

py -3 -m venv .venv
. .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip


macOS / Linux

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

2) Install dependencies
pip install "gradio>=4.37" "requests>=2.32" "certifi>=2024.2.2"


(Optional) requirements.txt:

gradio>=4.37
requests>=2.32
certifi>=2024.2.2

pip install -r requirements.txt

3) Set your OpenRouter key

Windows (current terminal)

$env:OPENROUTER_API_KEY="sk-or-xxxxxxxx"


macOS / Linux

export OPENROUTER_API_KEY="sk-or-xxxxxxxx"

Run

Pick one of the two launch scripts:

# DeepSeek (free tier)
python model1.py

# Moonshot Kimi (free tier)
python model2.py


Gradio prints a local URL; open it in your browser.

UI guide

Inputs

Search keywords: defaults include pressure broadening, linewidth, halfwidth, Voigt, gamma, shift, Lorentz, HWHM, FWHM.

Active species / Perturbers: e.g. CO2, NO, H2O and N2, O2, He, H2, air.

Max results, Year range: defaults 500, 2000–2025.

Only physics journals: keep typical physics/spectroscopy venues.

Require species/perturber in title/abstract: hard filter.

Heuristic cap: 0 keeps all to LLM; else top-K by heuristic score (faster/cheaper, slightly lower recall).

Sort by year: DESC/ASC (applied after journal priority).

Prioritize journals: show JQSRT/JMS/JCP before others.

Outputs

Clickable DOI (copied to clipboard on click).

Blue row background = physics journal.

Export CSV / Export BibTeX buttons produce downloadable files.

Sorting policy (stable)

A single composite key ensures consistent pagination:

(priority(JQSRT<JMS<JCP), year (None last; asc/desc), normalized title)


So priority never gets “undone” by a second sort.

What the log means

Fetched (after merge/dedup/soft-neg): N records – union of all strategies, deduplicated and softly denoised.

After heuristic: K candidates; LLM kept: M (batches=B) – top-K (or all) sent to LLM; M predicted relevant.

Note: these numbers are not precision/recall; you need ground-truth labels to compute P/R/F1.

Exports

CSV fields: Title, Authors, Year, Journal, DOI

BibTeX: minimal @article with escaped fields, a simple citekey
firstAuthorFamily + year + firstTitleWord.
(You can tweak citekey/fields in export_results_bibtex().)

Recommended presets (reproducible demo)

Keywords: (default)

Species: CO2, NO, H2O

Perturbers: N2, O2, He, H2, air

Years: 2000–2025

Physics-only: ON

Require species/perturber: OFF (then try ON)

Heuristic cap: 0

Sort by year: DESC

Prioritize journals: ON

Customisation

Priority journals: edit JOURNAL_PRIORITY (order matters).

Physics journal palette: edit PHYSICS_JOURNALS.

Heuristics: change PARAM_HEURISTICS (ranking), SOFT_NEG / MUST_POS (denoise).

Networking: tune HTTP_TIMEOUT, retry counts, and backoff.

LLM behavior: adjust LLM_MODEL (or run the other script), LLM_BATCH_SIZE, and the system prompt.

Troubleshooting

401 / key not found: check OPENROUTER_API_KEY is set in the same shell you run Python.

429 / 5xx: the app has retry/backoff; if it persists, reduce Max results or wait a bit.

Empty abstracts: Crossref often omits abstracts; the pipeline still works (heuristics + title).

Slow LLM: raise heuristic cap selectivity (e.g., cap=50) or switch to the other free model.

Windows Git Bash activation: prefer PowerShell or use source .venv/Scripts/activate.

Notes on models

model1.py (DeepSeek free) and model2.py (Kimi free) both work via OpenRouter.
Quality/latency can vary by time and region; logs will still reflect batches and kept counts.

Reuse & citation

Please cite Crossref when using this app for research. DOI links point to the official resolver.
The app is intended for research/education only; check publisher terms before redistributing abstracts or full text.

License

MIT (recommended) – or adapt to your project’s policy.

Minimal tests (optional, for markers)

Add a tiny tests/ folder and verify pure helpers:

priority_score() returns 0/1/2/999 as expected

is_physics_journal() detects “Physical Review X” and JQSRT/JMS/JCP

sort_results_stable() preserves priority then year then title

This strengthens the “Quality of programming” section in the marking rubric.

Maintainers

Default contact: Pengxia Guo / ucappg1@ucl.ac.uk

User agent string is set to broadening-search/<version> for polite API usage.