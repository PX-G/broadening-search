# Spectral Line Broadening Literature Search

Crossref multi-strategy retrieval + spectroscopy-aware heuristics + **batch LLM triage**, with **stable sorting** (Journal priority → Year → Title), optional **species/perturber hard filter**, and one-click **CSV / BibTeX export**.  
Built to quickly surface papers reporting **experimental pressure-broadening parameters** (γ, n, line shifts, HWHM/FWHM, Voigt/Lorentz).

> Use `model1.py` (DeepSeek) or `model2.py` (Moonshot Kimi). The two files are identical except for the `LLM_MODEL` line.

---

## Why this tool?

“Linewidth” shows up in many unrelated contexts (optical clocks, micro-resonators, HEP “gamma rays”…). This app:

1. casts a **wide Crossref net** across multiple fields/queries  
2. **deduplicates** results  
3. **soft-denoises** with domain terms  
4. **ranks heuristically** by spectroscopy keywords  
5. **batches an LLM** to keep likely experimental broadening papers  
6. **stably sorts** so JQSRT/JMS/JCP appear first, then year, then title

---

## Features

- Crossref search with **multi-strategy recall** (query / title / bibliographic)
- **Physics-journal gating** and **priority list** (JQSRT → JMS → JCP)
- **Species/perturber hard filter** (must appear in title/abstract)
- **Stable sort**: priority → year (asc/desc) → title (tie-break)
- **Retry/backoff** for HTTP/429/5xx
- Keyword **highlighting**, **click-to-copy DOI**
- **Export CSV** (`broadening_results.csv`) and **BibTeX** (`broadening_results.bib`)
- Clear **logs** (fetched / heuristic kept / LLM kept with batch count)

---

## Project structure
```
broadening-search/
  broadening-search/
    __init__.py
    model1.py
    model2.py
  tests/
    conftest.py
    test_utils.py
    test_sort.py
    test_export.py
    test_crossref_search_mock.py
  sample ouput/
    broadening_results.csv
    broadening_results.bib
    sample output interface.jpeg
=======


```


## Install

### 1) Create & activate a virtual environment

**Windows (PowerShell)**
```powershell
py -3 -m venv .venv
. .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

```

**MacOS / Linux (bash)**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```
### 2) Install dependencies
```bash
pip install "gradio>=4.37" "requests>=2.32" "certifi>=2024.2.2"
```

(Optional)`requirements.txt` :
```powershell
gradio>=4.37
requests>=2.32
certifi>=2024.2.2
```
```bash
pip install -r requirements.txt
```



### 3) Set your OpenRouter key

**Windows (current terminal)**
```powershell
$env:OPENROUTER_API_KEY="sk-or-xxxxxxxx"
```

**macOS / Linux**
```bash
export OPENROUTER_API_KEY="sk-or-xxxxxxxx"
```
## Run

Pick one of the two launch scripts(bash):
```bash
# DeepSeek (free tier)
python broadening_search\\model1.py

# Moonshot Kimi (free tier)
python broadening_search\\model2.py
```
Gradio will print a local URL; open it in your browser.

## UI guide

**Inputs**

- Search keywords – defaults include pressure broadening, linewidth, halfwidth, Voigt, gamma, shift, Lorentz, HWHM, FWHM.

- Active species / Perturbers – e.g. CO2, NO, H2O and N2, O2, He, H2, air.

- Max results / Year range – defaults 500, 2000–2025.

- Only physics journals – keep typical physics/spectroscopy venues.

- Require species/perturber in title/abstract – hard filter.

- Heuristic cap – 0 keeps all to LLM; otherwise send only top-K by heuristic score (faster/cheaper).

- Sort by year – DESC/ASC (applied after journal priority).

- Prioritize journals – show JQSRT/JMS/JCP first.

**Outputs**

- Clickable DOI (also copies DOI to clipboard).

- Blue row = physics journal.

- Export CSV / Export BibTeX produce downloadable files.


  
### What the log means

- **Fetched (after merge/dedup/soft-neg): N records** — union of all strategies, deduplicated and softly denoised.  
- **After heuristic: K candidates; LLM kept: M (batches=B)** — top-K (or all) sent to LLM; M predicted relevant.

> These numbers are **not** precision/recall; computing **P/R/F1** requires ground-truth labels.

---

### Exports

- **CSV** columns: `Title, Authors, Year, Journal, DOI`  
- **BibTeX**: minimal `@article` entries with escaped fields and a simple citekey  
  `firstAuthorFamily + year + firstTitleWord`.  
  *(Adjust in `export_results_bibtex()` if you prefer another scheme.)*

---

### Recommended demo preset (reproducible)

- **Keywords:** *(default)*  
- **Species:** `CO2, NO, H2O`  
- **Perturbers:** `N2, O2, He, H2, air`  
- **Years:** `2000–2025`  
- **Physics-only:** **ON**  
- **Require species/perturber:** **OFF** *(then try **ON**)*  
- **Heuristic cap:** `0`  
- **Sort by year:** `DESC`  
- **Prioritize journals:** **ON**

---

### Customisation

- **Priority journals** — edit `JOURNAL_PRIORITY` *(order matters)*.  
- **Physics list** — edit `PHYSICS_JOURNALS`.  
- **Heuristics** — tweak `PARAM_HEURISTICS`, `SOFT_NEG`, `MUST_POS`.  
- **Networking** — tune timeouts/retries in the config block.  
- **LLM** — switch between `model1.py` and `model2.py`, adjust batch size and prompt.

---

### Troubleshooting

- **401 / key not found** — ensure `OPENROUTER_API_KEY` is set in the **same shell**.  
- **429 / 5xx** — built-in retry/backoff; reduce *Max results* or rerun later if needed.  
- **Empty abstracts** — Crossref often omits them; pipeline still works via title + heuristics.  
- **Slow LLM** — use a smaller heuristic cap (e.g., `50`) or try the other model file.  
- **Windows Git Bash** — prefer PowerShell to activate venv, or `source .venv/Scripts/activate`.


---

## Testing

This project ships with a small offline test suite (no network calls) using `pytest`.

### Quick start
```bash
# 1) Activate your venv
# Windows (PowerShell)
. .\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# 2) Install test dependency
pip install pytest

# 3) Set a dummy key so the module can be imported during tests
# Windows (PowerShell)
$env:OPENROUTER_API_KEY = "test"
# macOS / Linux
export OPENROUTER_API_KEY="test"

# 4) Run tests
python -m pytest -q
```
### Model parity (model1.py vs model2.py)

`model1.py` and `model2.py` are identical **except** for the `LLM_MODEL` setting:

- `model1.py`: `deepseek/deepseek-chat-v3-0324:free`  
- `model2.py`: `moonshotai/kimi-k2:free`

All retrieval, denoising, sorting, exporting, and UI logic is shared. The unit tests target **model-agnostic** functionality (Crossref query building, soft negatives, deduplication, stable sorting, CSV/BibTeX export), so we only ship tests for `model1.py`; the results apply equally to `model2.py`. Switching models does not require any test changes. *(Note: End-to-end LLM triage may yield different paper sets, but this does not affect the tested invariants.)*

## Offline dataset for reproducibility

We provide the exact items exported from the app so reviewers can replicate results **without internet access**.

- **CSV**: [`sample output/broadening_results.csv`](sample output/broadening_results.csv)  
  Columns: `Title, Authors, Year, Journal, DOI`
- **BibTeX**: [`sample output/broadening_results.bib`](sample output/broadening_results.bib)  
  Minimal `@article` entries; citekey = `firstAuthorFamily + year + firstTitleWord`.

**Provenance (UI settings):**

- Keywords: `pressure broadening, linewidth, halfwidth, Voigt, gamma, shift, Lorentz, HWHM, FWHM`
- Species: `H2O`
- Perturbers: `N2, O2, He, H2, air`
- Years: `2000–2025`
- Physics-only: **ON**
- Require species/perturber in title/abstract: **OFF**
- Heuristic cap: `0` (keep all)
- Sort by year: `DESC (newest first)`
- Prioritize journals (JQSRT → JMS → JCP): **ON**

> **Note:** `model1.py` and `model2.py` are identical apart from the LLM model ID.  
> The dataset schema is the same for both; the exported CSV/BIB are model-agnostic.  
> The **sample** files provided here were generated with `model1.py` (`deepseek/deepseek-chat-v3-0324:free`); minor differences may occur when re-running or when using `model2.py` due to model/version/serving variability and non-determinism.


=======
### Testing

This project ships with a small offline test suite (no network calls) using `pytest`.

#### Quick start
```bash
# 1) Activate your venv
# Windows (PowerShell)
. .\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# 2) Install test dependency
pip install pytest

# 3) Set a dummy key so the module can be imported during tests
# Windows (PowerShell)
$env:OPENROUTER_API_KEY = "test"
# macOS / Linux
export OPENROUTER_API_KEY="test"

# 4) Run tests
python -m pytest -q
```
#### Model parity (model1.py vs model2.py)

`model1.py` and `model2.py` are identical **except** for the `LLM_MODEL` setting:

- `model1.py`: `deepseek/deepseek-chat-v3-0324:free`  
- `model2.py`: `moonshotai/kimi-k2:free`

All retrieval, denoising, sorting, exporting, and UI logic is shared. The unit tests target **model-agnostic** functionality (Crossref query building, soft negatives, deduplication, stable sorting, CSV/BibTeX export), so we only ship tests for `model1.py`; the results apply equally to `model2.py`. Switching models does not require any test changes. *(Note: End-to-end LLM triage may yield different paper sets, but this does not affect the tested invariants.)*


**Maintainer:** *Pengxia Guo / ucappg1@ucl.ac.uk*  
User-agent is `broadening-search/<version>` for polite API usage.
