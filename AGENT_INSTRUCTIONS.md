# Project Context: GCMS-App
You are an expert in Python, Cheminformatics, Mass Spectrometry (LC-MS/GC-MS), and Data Science. This project is an automated HPLC-GCMS pipeline that processes `.mzML` files into an AI-verified chromatogram report.

## Core Libraries & Tools
- Mass Spec Parsing: `pyteomics`, `psims`
- Peak Detection & Fitting: `scipy.signal`, `hplc-py`
- Cheminformatics & Fragmentation: `rdkit`, `mendeleev`, `pubchempy`
- AI Integration: `google.genai` (Gemini API)
- UI/Visualization: `ipyketcher`, `matplotlib`, `mols2grid`, `ipywidgets`

## Architectural Rules & Constraints
1. **Peak Detection is Strictly Numerical:** Do NOT use AI or Gemini for detecting peaks. Peak detection must remain purely numerical, using `scipy.signal.find_peaks` and the `hplc-py` baseline correction/Gaussian deconvolution engines.
2. **AI's Specific Role:** Gemini is only used downstream to:
   - Verify/reject candidate compound assignments based on adduct and fragment evidence.
   - Propose identities for peaks lacking library matches.
   - Power the interactive chat session about the results.
3. **Library Matching:** RDKit is used for in-silico fragmentation trees. Match observed spectra numerically within a strict `TOLERANCE_DA = 0.05` limit.
4. **Environment:** The code is designed to run in Google Colab/Jupyter Notebook environments. Ensure any UI elements utilize `ipywidgets` or `ipyketcher` appropriately.

## Coding Style
- Write clean, documented Python code with type hinting where possible.
- When modifying the peak fitting hyperparameter optimization, ensure the elbow/L-curve test logic remains intact.
- Handle API rate limits and potential 503 errors gracefully (especially for Gemini and NCI CACTUS queries).
