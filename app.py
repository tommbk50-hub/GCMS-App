import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import zipfile
import io

# --- 1. UI Setup ---
st.set_page_config(page_title="Virtual Combinatorial Library", layout="wide")
st.title("💊 Virtual Combinatorial Library Generator")

st.markdown("""
This app replicates the ZINC22 decoration pipeline. 
**Step 1:** Define your Core. **Step 2:** Upload Reactants (Fragments). **Step 3:** Enumerate.
""")

# --- 2. Input: Core Molecule ---
st.sidebar.header("1. Core Definition")
core_smiles = st.sidebar.text_input("Core SMILES", value="O=C(O)[C@@H]1CNC[C@H]1c2cccc(F)c2")

# Visualize the core
if core_smiles:
    mol = Chem.MolFromSmiles(core_smiles)
    if mol:
        st.sidebar.image(Chem.Draw.MolToImage(mol), caption="Core Structure")
    else:
        st.sidebar.error("Invalid SMILES")

# --- 3. Input: Fragments (The "ZINC" part) ---
# NOTE: In a web app, downloading 50GB is impossible. 
# We let the user upload a cleaned .smi file or use a demo set.
st.sidebar.header("2. Fragments Source")
uploaded_file = st.sidebar.file_uploader("Upload Fragments (.smi or .txt)", type=["smi", "txt"])

fragment_smiles = []
if uploaded_file:
    # Read uploaded file
    stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
    for line in stringio:
        parts = line.strip().split()
        if parts:
            fragment_smiles.append(parts[0])
    st.sidebar.success(f"Loaded {len(fragment_smiles)} fragments.")
else:
    st.sidebar.info("Upload a file to begin.")

# --- 4. Reaction Definitions (Simplified for Demo) ---
# You would paste your full dictionary from the notebook here
RXN_DEFINITIONS = {
    "Amide Coupling": AllChem.ReactionFromSmarts("[N;R;H1:2].[C:1](=O)[O;H1]>>[C:1](=O)[N:2]"),
    "SNAr": AllChem.ReactionFromSmarts("[N;R;H1:2].[c:1]F>>[c:1][N:2]")
}

rxn_choice = st.selectbox("Select Reaction Type", list(RXN_DEFINITIONS.keys()))

# --- 5. The "Run" Button ---
if st.button("Generate Library"):
    if not fragment_smiles:
        st.error("Please upload fragments first.")
    else:
        results = []
        core_mol = Chem.MolFromSmiles(core_smiles)
        rxn = RXN_DEFINITIONS[rxn_choice]
        
        progress_bar = st.progress(0)
        
        # Enumerate (Simplified loop)
        for i, frag_smi in enumerate(fragment_smiles):
            frag_mol = Chem.MolFromSmiles(frag_smi)
            if frag_mol:
                try:
                    # Run reaction
                    products = rxn.RunReactants((core_mol, frag_mol))
                    for p in products:
                        p_mol = p[0]
                        Chem.SanitizeMol(p_mol)
                        results.append(Chem.MolToSmiles(p_mol))
                except:
                    pass
            
            # Update progress every 10%
            if i % max(1, len(fragment_smiles)//10) == 0:
                progress_bar.progress((i + 1) / len(fragment_smiles))
        
        progress_bar.progress(100)
        
        # --- 6. Results & Download ---
        st.success(f"Generated {len(results)} Unique Products!")
        
        if results:
            df_results = pd.DataFrame(results, columns=["SMILES"])
            st.dataframe(df_results.head(50)) # Preview
            
            # Download Button
            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results CSV", csv, "library.csv", "text/csv")
