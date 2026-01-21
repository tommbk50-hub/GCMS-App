import streamlit as st
import sys
import io
import re
import base64
import itertools
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpld3
from mpld3 import plugins
from matplotlib import cm
from matplotlib import patheffects as pe
from adjustText import adjust_text

# RDKit Imports
from rdkit import Chem
from rdkit.Chem import Recap, Descriptors, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdMolDescriptors as rdMD
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="GCMS Simulator",
    page_icon="⚗️",
    layout="wide"
)

st.title("⚗️ Mass Spectrometry Fragmentation Tree & GCMS Simulator")
st.markdown("""
This app simulates GCMS spectra and fragmentation trees from SMILES strings. 
based on the *GCMS_simulator* notebook.
""")

# -----------------------------------------------------------------------------
# SIDEBAR INPUTS
# -----------------------------------------------------------------------------
st.sidebar.header("Configuration")

default_smiles = "OC[C@H]1[C@H](C2=CC(F)=CC=C2)CN(C3=CC(C)=NC=C3C#N)C1"
smiles_input = st.sidebar.text_area("SMILES String", value=default_smiles, height=100)

st.sidebar.subheader("Spectrum Settings")
ion_mode = st.sidebar.selectbox("Ion Mode", ["ESI+", "ESI-", "EI"], index=0)
preset_mode = st.sidebar.selectbox("Visual Preset", 
    ["publication", "crowded", "lightweight", "presentation", "parent_focus"], 
    index=3
)
label_mode = st.sidebar.selectbox("Label Mode", ["ion+mz", "ion", "mz"], index=0)

st.sidebar.subheader("Tree Settings")
tree_mode = st.sidebar.radio("Fragmentation Mode", ["MS-like (hetero/α/ring)", "Retrosynthetic (RECAP)"])

run_btn = st.sidebar.button("Generate Simulation", type="primary")

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS (Refactored from Notebook)
# -----------------------------------------------------------------------------

def _smiles2mol(smi, add_hs=False):
    m = Chem.MolFromSmiles(smi)
    if m:
        return Chem.AddHs(m) if add_hs else m
    return None

def _ion_mass(mol):
    return Descriptors.ExactMolWt(mol) + 1.007276, 1

# --- Recap Fragmentation ---
def _recap_levels(mol):
    if not mol: return []
    root = Recap.RecapDecompose(mol)
    roots = [root] if root.smiles else list(root.children.values())
    levels = []
    def walk(node, d=0):
        if len(levels) <= d:
            levels.append([])
        levels[d].append(node.smiles)
        for ch in node.children.values():
            walk(ch, d+1)
    for n in roots: walk(n, 0)
    return levels

# --- MS-style Fragmentation ---
def _cleavable_bonds(mol):
    hetero = lambda a: a.GetAtomicNum() not in (1, 6)
    cleav = set()
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE: continue
        i, j = bond.GetBeginAtom(), bond.GetEndAtom()
        if hetero(i) or hetero(j):
            cleav.add(bond.GetIdx()); continue
        alpha = lambda at: any(hetero(n) for n in at.GetNeighbors())
        if alpha(i) or alpha(j):
            cleav.add(bond.GetIdx()); continue
        if bond.IsInRing():
            cleav.add(bond.GetIdx())
    return list(cleav)

def _ms_levels(mol, max_depth=3, min_heavy=2):
    if not mol: return []
    levels, frontier = [[Chem.MolToSmiles(mol)]], [mol]
    seen = set(levels[0])
    for _ in range(max_depth):
        nxt = []
        for m in frontier:
            for b in _cleavable_bonds(m):
                frag = Chem.FragmentOnBonds(m, [b], addDummies=False)
                parts = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=True)
                for p in parts:
                    if p.GetNumHeavyAtoms() < min_heavy: continue
                    smi = Chem.MolToSmiles(p)
                    if smi not in seen:
                        seen.add(smi); nxt.append(p)
        if not nxt: break
        levels.append([Chem.MolToSmiles(x) for x in nxt])
        frontier = nxt
    return levels

# --- Draw Tree ---
def get_tree_svg(smiles, mode_str):
    mol = _smiles2mol(smiles)
    if not mol: return None
    
    if "RECAP" in mode_str:
        levels = _recap_levels(mol)
    else:
        levels = _ms_levels(mol)

    if not levels: return None

    n_cols = max(len(row) for row in levels)
    mols, legs = [], []
    for smi in itertools.chain.from_iterable(levels):
        m = _smiles2mol(smi)
        mz, _ = _ion_mass(m)
        mols.append(m)
        legs.append(f"m/z={mz:.2f}")

    pad = n_cols * len(levels) - len(mols)
    mols.extend([Chem.Mol()] * pad)
    legs.extend([""] * pad)

    w_cell, h_cell = 260, 260
    drawer = rdMolDraw2D.MolDraw2DSVG(
        n_cols*w_cell, (len(mols)//n_cols)*h_cell if n_cols else h_cell, w_cell, h_cell)
    drawer.drawOptions().legendFontSize = 18
    drawer.DrawMolecules(tuple(mols), legends=tuple(legs))
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    
    # White background injection
    svg = re.sub(r'<svg[^>]*>', lambda m: m.group(0) + '<rect width="100%" height="100%" fill="white"/>', svg, count=1)
    svg = svg.replace('<?xml version="1.0" encoding="UTF-8"?>', '')
    return svg

# --- Simulation Constants & Data ---
PROTON = 1.007276466812
ADDUCT_MASSES = {
    "H+": PROTON, "NH4+": 18.033823, "Na+": 22.989218, "K+": 38.963158, "[M]•+": 0.0,
    "H-": -PROTON, "Cl-": 34.969402, "CH3COO-": 59.013851, "HCOO-": 44.998201, "NO3-": 61.987819
}
ADDUCT_WEIGHTS = {"H+":1.0,"NH4+":0.6,"Na+":0.4,"K+":0.25,"H-":1.0,"Cl-":0.4,
                  "CH3COO-":0.3,"HCOO-":0.35,"NO3-":0.2,"[M]•+":1.0}
ALPHA = 0.6
ADDUCT_CATALOG = {
    "ESI+": [
        ("[M+H]+",          PROTON,                    +1, 62.6, 1),
        ("[M+NH4]+",        18.033823,                 +1, 14.4, 1),
        ("[M+Na]+",         22.989218,                 +1,  8.8, 1),
        ("[M+K]+",          38.963158,                 +1,  3.2, 1),
        ("[M+2H]2+",        2*PROTON,                  +2,  2.0, 1),
        ("[2M+H]+",         PROTON,                    +1,  2.0,  2),
    ],
    "ESI-": [
        ("[M-H]-",          -PROTON,                   -1, 50.0, 1),
        ("[M+Cl]-",         34.969402,                 -1,  8.0, 1),
        ("[M+FA-H]-",       44.998201,                 -1, 10.0, 1),
    ],
    "EI": [("[M]•+", 0.0, +1, 100.0, 1)],
}

PRESETS = {
    "publication": {"FIG_W_IN":18.0, "FIG_H_IN":6.5, "FIG_DPI":160, "SAMPLES_PER_STICK":28, "LABEL_FONTSIZE":12},
    "crowded": {"FIG_W_IN":22.0, "FIG_H_IN":8.0, "FIG_DPI":160, "SAMPLES_PER_STICK":30, "LABEL_FONTSIZE":12},
    "lightweight": {"FIG_W_IN":16.0, "FIG_H_IN":6.0, "FIG_DPI":120, "SAMPLES_PER_STICK":18, "LABEL_FONTSIZE":11},
    "presentation": {"FIG_W_IN":24.0, "FIG_H_IN":9.0, "FIG_DPI":150, "SAMPLES_PER_STICK":30, "LABEL_FONTSIZE":14},
    "parent_focus": {"FIG_W_IN":18.0, "FIG_H_IN":6.5, "FIG_DPI":150, "SAMPLES_PER_STICK":26, "LABEL_FONTSIZE":12},
}

def _frag_weight(frag, adduct):
    mass = rdMD.CalcExactMolWt(frag)
    hetero = sum(1 for a in frag.GetAtoms() if a.GetAtomicNum() not in (1,6))
    return (hetero+1) * mass**(-ALPHA) * ADDUCT_WEIGHTS.get(adduct, 0.2)

def _molecular_adducts(mass_M, mode):
    out = []
    cat = ADDUCT_CATALOG.get(mode, [])
    # Fallback if mode not in simplified catalog
    if not cat: cat = ADDUCT_CATALOG["ESI+"]
    for name, shift, z, prob, mMult in cat:
        z_abs = abs(int(z)) if int(z) != 0 else 1
        mz = (mMult*mass_M + shift) / z_abs
        out.append((name, round(mz, 4), float(prob)))
    return out

def simulate_spectrum(smiles, mode):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    pmass = rdMD.CalcExactMolWt(mol)

    if mode == "ESI+": frag_adducts = ("H+","NH4+","Na+","K+")
    elif mode == "ESI-": frag_adducts = ("H-","Cl-","CH3COO-","HCOO-","NO3-")
    else: frag_adducts = ("[M]•+",)

    from collections import defaultdict, Counter
    label_map = defaultdict(list)
    repr_frag = {}
    records = []

    # 1) Fragments
    w_frag = Counter()
    for b in mol.GetBonds():
        if b.IsInRing() or b.GetBondType()!=Chem.BondType.SINGLE: continue
        split = Chem.FragmentOnBonds(mol,[b.GetIdx()], addDummies=False)
        for frag in Chem.GetMolFrags(split, asMols=True, sanitizeFrags=True):
            if frag.GetNumHeavyAtoms()<2: continue
            mass = rdMD.CalcExactMolWt(frag)
            formula = CalcMolFormula(frag)
            for ad in frag_adducts:
                mz = round(mass + ADDUCT_MASSES[ad], 4)
                w = _frag_weight(frag, ad)
                w_frag[mz] += w
                lab = f"{formula}.{ad}"
                label_map[mz].append(lab)
                records.append({"mz":mz,"mol":frag,"label":lab,"w":w})
                if (mz not in repr_frag) or w>repr_frag[mz][2]:
                    repr_frag[mz]=(frag,lab,w)

    if w_frag:
        fmax = max(w_frag.values())
        scale = 100.0 / fmax if fmax > 0 else 1.0
        for m in list(w_frag): w_frag[m] *= scale

    # 2) Adducts
    w_add = Counter()
    prob_by_mz = {}
    adduct_name_by_mz = {}
    for name, mz, prob in _molecular_adducts(pmass, mode):
        w = max(prob, 0.1)
        w_add[mz] += w
        label_map[mz].append(name)
        records.append({"mz":mz,"mol":mol,"label":name,"w":w})
        if (mz not in repr_frag) or w>repr_frag[mz][2]:
            repr_frag[mz]=(mol,name,w)
        prob_by_mz[mz] = prob
        adduct_name_by_mz[mz] = name

    if w_add:
        amax = max(w_add.values())
        scale = 80.0 / amax if amax > 0 else 1.0
        for m in list(w_add): w_add[m] *= scale

    combined = Counter(); combined.update(w_frag); combined.update(w_add)
    vals = list(combined.values())
    base = max(vals) if vals else 1e-12
    peaks = {}
    for m, w in combined.items():
        ri = 100.0 * (w / base) ** 0.65 # Gamma
        if ri > 2.0: peaks[m] = int(round(ri))

    ion_labels = {m:", ".join(sorted(set(v))) for m,v in label_map.items()}
    repr_map = {m:{"mol":t[0], "label":t[1]} for m,t in repr_frag.items()}
    
    # Colors
    cols_by_mz = {}
    if prob_by_mz:
        norm = mpl.colors.Normalize(vmin=0, vmax=65, clip=True)
        cmap = cm.get_cmap("RdYlBu_r")
        for mz, p in prob_by_mz.items():
            cols_by_mz[mz] = cmap(norm(p))
            
    return peaks, ion_labels, repr_map, records, cols_by_mz, prob_by_mz

def _thumb_png_b64(mol):
    pil = Draw.MolToImage(mol, size=(200,150), kekulize=True)
    buf = io.BytesIO(); pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

# -----------------------------------------------------------------------------
# MAIN APP LOGIC
# -----------------------------------------------------------------------------

if run_btn and smiles_input:
    st.divider()
    
    # --- Tab 1: Fragmentation Tree ---
    tab1, tab2, tab3 = st.tabs(["Fragmentation Tree", "Simulated Spectrum", "Data Table"])
    
    with tab1:
        with st.spinner("Generating Fragmentation Tree..."):
            try:
                svg_data = get_tree_svg(smiles_input, tree_mode)
                if svg_data:
                    # Encode SVG to base64 for display
                    b64 = base64.b64encode(svg_data.encode('utf-8')).decode("utf-8")
                    html = f'<img src="data:image/svg+xml;base64,{b64}" style="width:100%; max-width:1000px;"/>'
                    st.markdown(html, unsafe_allow_html=True)
                else:
                    st.warning("Could not generate tree for this molecule.")
            except Exception as e:
                st.error(f"Error in Tree Generation: {e}")

    # --- Tab 2: Spectrum ---
    with tab2:
        with st.spinner("Simulating Spectrum..."):
            try:
                result = simulate_spectrum(smiles_input, ion_mode)
            except Exception as e:
                result = None
                st.error(f"Simulation failed: {e}")

            if result:
                peaks, ion_labels, repr_map, records, col_map, prob_map = result
                
                # Matplotlib Plotting
                cfg = PRESETS[preset_mode]
                mz_arr = np.array(sorted(peaks), float)
                int_arr = np.array([peaks[m] for m in mz_arr], float)

                fig, ax = plt.subplots(figsize=(cfg["FIG_W_IN"], cfg["FIG_H_IN"]), dpi=cfg["FIG_DPI"])
                ax.set_facecolor("white"); fig.patch.set_facecolor("white")
                ax.vlines(mz_arr, 0, int_arr, lw=2, color="#1f77b4")
                ax.set_xlabel("m/z"); ax.set_ylabel("Relative Intensity")
                ax.spines[["top","right"]].set_visible(False)
                ax.set_xlim(mz_arr.min()-5, mz_arr.max()+5)
                ax.set_ylim(0, max(int_arr)*1.25)
                
                # Labels
                texts = []
                # Simple labeling logic for Streamlit performance
                # (Complex collision detection from notebook simplified slightly for speed)
                sorted_peaks = sorted(zip(mz_arr, int_arr), key=lambda x: x[1], reverse=True)
                for m, y in sorted_peaks[:cfg.get("MAX_LABELS", 50)]:
                    lbl = ion_labels.get(m, "").split(",")[0]
                    if label_mode == "mz": lbl = f"{m:.2f}"
                    elif label_mode == "ion+mz": lbl = f"{lbl}\n{m:.2f}"
                    
                    c = col_map.get(m, "black")
                    t = ax.text(m, y, lbl, ha="center", va="bottom", rotation=90, color=c, fontsize=9)
                    texts.append(t)
                
                try:
                    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))
                except: pass

                # Interactive Tooltips (mpld3)
                tooltips = []
                xs, ys = [], []
                for m, y in zip(mz_arr, int_arr):
                    # Invisible scatter points for tooltips
                    xs.append(m); ys.append(y/2) 
                    
                    info = repr_map.get(m, {})
                    mol_obj = info.get("mol")
                    cap = info.get("label", f"{m:.2f}")
                    
                    if mol_obj:
                        b64_img = _thumb_png_b64(mol_obj)
                        tt = f"""<div style="background:white; padding:10px; border-radius:5px; border:1px solid #ddd;">
                                 <img src="data:image/png;base64,{b64_img}"/><br>
                                 <b>{cap}</b><br>m/z: {m:.4f}</div>"""
                    else:
                        tt = f"<div><b>{cap}</b><br>m/z: {m:.4f}</div>"
                    tooltips.append(tt)

                points = ax.scatter(mz_arr, int_arr, s=50, alpha=0.01) # Invisible triggers
                plugins.connect(fig, plugins.PointHTMLTooltip(points, tooltips, css=""))

                # Render using Streamlit Component
                fig_html = mpld3.fig_to_html(fig)
                st.components.v1.html(fig_html, height=800, scrolling=True)
                plt.close(fig)

    # --- Tab 3: Data ---
    with tab3:
        if 'records' in locals():
            data_rows = []
            for r in records:
                m = r['mol']
                s = Chem.MolToSmiles(m) if m else ""
                data_rows.append({"SMILES": s, "Peak Label": r['label'], "m/z": r['mz'], "Weight": r['w']})
            
            df = pd.DataFrame(data_rows)
            st.dataframe(df, use_container_width=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", data=csv, file_name="spectrum_data.csv", mime="text/csv")

elif not smiles_input:
    st.info("Please enter a SMILES string in the sidebar to begin.")
