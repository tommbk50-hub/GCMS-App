# Automated HPLC-GCMS Python Notebook using AI.

## `auto_hplc_gcms (version 1).py` — Summary

A Colab-exported notebook (8 cells) that turns a `.mzML` LC‑MS file into an annotated, AI‑verified chromatogram report.

## How packages + Gemini combine for automatic peak detection

- **`pyteomics.mzml`** streams the `.mzML` file, keeping MS1 scans only, and pulls `scan start time`, `total ion current`, `m/z array` and `intensity array` → a pandas `Time`/`Signal` TIC chromatogram.
- **`scipy.signal.find_peaks` / `peak_prominences`** finds raw topographical apices and their left/right baseline boundaries on a min–max normalised signal.
- **`hplc-py`** (`hplc.quant.Chromatogram.fit_peaks`, cloned from GitHub) does the real chemistry: baseline correction (SNIP), windowing (`time_window=[200, 300]`) and Gaussian/skew‑normal mixture deconvolution, returning retention time, area, width per peak.
- **Prominence is chosen automatically**, not hard-coded: the script sweeps 15 log-spaced prominence values, records the peak count for each, then applies an **elbow/L‑curve test** (maximum perpendicular distance from the secant line of the descending branch) to pick the optimal threshold. An `ipywidgets` box lets you override it.
- **Gemini is *not* used for detecting peaks** — detection is purely numerical (scipy + hplc-py). Gemini (`google.genai`, auto-selecting the newest `gemini-X.Y-flash` model) is used **downstream** to (a) verify or reject each candidate compound assignment given intact-adduct and fragment evidence, (b) propose an identity for peaks with no library match, and (c) power a chat session about the results.
- **RDKit** supplies the "library": in silico fragmentation trees + adduct/isotope m/z tables that are matched numerically to the observed spectra within `TOLERANCE_DA = 0.05`; Gemini only arbitrates the final call.

## Cell-by-cell breakdown

**1. Install required Python libraries** (L10–19)
- `pip install` ipyketcher, rdkit, mols2grid, pyteomics, psims, lxml, hdf5plugin, pandas, pubchempy, mendeleev, marimo, wigglystuff.
- `git clone` the `cremerlab/hplc-py` repo for the deconvolution engine.

**2. Plot HPLC Chromatogram → `hplc_df`** (L21–71)
- Downloads `FeatureFinderMetaboIdent_1_input.mzML` from the OpenMS test suite via `urllib`.
- Parses MS1 scans with `pyteomics.mzml`; builds `hplc_df` with `Signal` (TIC) and `Time` (RT, seconds).
- Plots the TIC trace with matplotlib.

**3. Automatically detect and fit the peaks → `detected_df`** (L73–370)
- Min–max normalises the signal; suppresses warnings.
- **Hyperparameter optimisation:** sweeps `np.logspace(-3,-1,15)` prominences through `fit_peaks`, applies the elbow method to pick `optimal_prominence`, exports it as `prominence_val`, and renders an L‑curve plot plus a table with the optimum row outlined in red.
- Builds an `ipywidgets` UI (prominence `FloatText` + "Rerun Peak Fitting" button).
- `run_peak_fitting()`: scipy prominence detection → start/end times; `hplc-py` deconvolution at the same prominence; maps each fitted Gaussian centre to its nearest raw apex to attach `peak start time`, `peak end time`, `topographical_apex`.
- Adds `normalised area` (% of total).
- Re-reads the `.mzML`, sums intensities per rounded m/z inside each peak's RT window, and stores the **top 10 dominant m/z** (≥2 % relative intensity) per peak.
- Displays a colour-coded styled table and an overlaid deconvolution plot; result stored globally as `detected_df`.

**4. Mass Spectrometry Fragmentation Tree → `df_tree`** (L372–887)
- Monkey-patches an `ipyketcher` traitlets bug.
- Hard-codes ~50 **adducts** (positive/negative, multimers, charge states) and a **stable-isotope table** (avoids repeated `mendeleev` lookups).
- Four fragmentation modes via RDKit: **RECAP**, **BRICS**, an **MS-like** rule set (cleaves single bonds at/α to heteroatoms and ring bonds, depth 3), and import from a CSV.
- `draw_frag_tree()` enumerates fragments, computes exact masses (`Descriptors.ExactMolWt`), formulas (`CalcMolFormula`), applies every adduct shift (with `(exact_mass*mult + shift)/charge`) and adds isotopologue peaks with estimated relative abundances; renders an SVG grid of structures labelled with m/z.
- UI: **PubChem search** (`pubchempy`, names/CAS/SMILES) plus a **Ketcher** drawing pad to add custom targets, a scrollable SVG gallery, and mode/ion-mode dropdowns.
- Outputs `df_tree` (Target number, SMILES, Peak Name, Ion Mode, Charge, m/z, Relative Abundance) and saves `frag_tree.csv`.

**5. Assign, Verify & Refine Mass Spectra Signals (AI loop)** (L889–~1575)
- Requires `detected_df`, `df_tree`, `hplc_df`; prompts for a Gemini API key via `getpass`; lists models and auto-selects the latest `*-flash`.
- Caches all MS1 scans in RAM (`global_mzml_cache`) and aggregates a full spectrum per peak window; filters noise below 1 % of max.
- For each peak (16 threads, `concurrent.futures.ThreadPoolExecutor`): matches observed m/z against `df_tree` within 0.05 Da, splitting hits into **intact** vs **fragment** matches, and scores each candidate target — intensity-weighted, with a large bonus for primary adducts (`[M+H]+`, `[M-H]-`, `[M+Na]+`) and a small bonus per supporting fragment.
- The winning target plus its anchor/fragment evidence is sent to **Gemini** (temperature 0.1) with a strict prompt returning `VERIFIED` / `REJECTED` + a one-line rationale.
- Peaks with **no** library match get a second Gemini prompt proposing a contaminant/background ion class (`AI PROPOSED`). API 503s are caught and reported.
- Produces `assigned_df` and annotated chromatogram / mass-spectra figures with embedded RDKit structure images.

**6. Automated Structural Verification (PubChem & NCI CACTUS)** (L1577–1721)
- Takes `VERIFIED` rows with primary, non-isotopic adducts; strips isotopes/explicit Hs but **keeps stereochemistry**.
- Generates a stereo-specific **InChIKey** with RDKit → queries the PubChem PUG REST synonyms endpoint; a heuristic filter picks the most "chemical-looking" common name (biochemical suffixes preferred).
- Queries **NCI CACTUS** for the official IUPAC name from isomeric SMILES; rate-limits at 0.5 s.
- Displays `gt_df` with colour-highlighted hits.

**7. 💬 Discuss your LC-MS Results with Gemini** (L1722–1872)
- Compiles a plain-text run summary (peak #, RT, normalised area, status, adduct, rationale) into a Gemini **system instruction** (no LaTeX, chemistry-expert persona).
- Opens a `client.chats.create` session (temperature 0.3) behind an `ipywidgets` chat UI with history re-rendering, Enter-to-send, and two quick prompts ("Summarize Findings", "Analyze Rejected Peaks").

**8. 📄 Export Results to HTML Report** (L1874–1973)
- Base64-embeds matplotlib figures and `to_html()` tables into a styled standalone document containing: labelled chromatogram, structural verification table, final assignments + AI rationale, mass-spectra grids, and the AI run summary context.
- Writes `lcms_analysis_report.html` and renders a Colab download link.
