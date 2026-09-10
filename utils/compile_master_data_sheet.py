import os
import pandas as pd
import numpy as np

def generate_master_sheet():
    verkerk_file = "../tlu/BT_results_summary.txt"
    run_20tree_file = "../output/GPGLMM_01_20tree.xlsx"
    run_100tree_file = "../output/GPGLMM_02_100tree.xlsx"
    output_master = "../output/Results_combined_BT_GPGLMM.xlsx"

    # verify workspace files are available
    missing = [f for f in [verkerk_file, run_20tree_file, run_100tree_file] if not os.path.exists(f)]
    if missing:
        print(f" Error: Missing required files in workspace: {missing}")
        return

    # load datasets
    df_v = pd.read_csv(verkerk_file, sep="\t")
    df_20 = pd.read_excel(run_20tree_file)
    df_100 = pd.read_excel(run_100tree_file)

    # standardize primary identifying headers
    df_v.rename(columns={"code": "Feature_ID"}, inplace=True)
    df_20.rename(columns={df_20.columns[0]: "Feature_ID"}, inplace=True)
    df_100.rename(columns={df_100.columns[0]: "Feature_ID"}, inplace=True)

    # normalize character strings to bypass casing mismatches
    for df in [df_v, df_20, df_100]:
        df["Feature_ID"] = df["Feature_ID"].astype(str).str.strip().str.lower()
        df.set_index("Feature_ID", inplace=True)

    master_records = {}

    # the 20-tree sheet identifies all 191 features
    for feat in df_20.index:
        row_20 = df_20.loc[feat]

        # extract 20-tree metrics
        status_20 = str(row_20.get("Status", "Analyzed"))
        beta_20 = float(row_20.get("GPGLMM_Param.", 0.0)) if status_20 == "Analyzed" else np.nan
        se_20 = float(row_20.get("GPGLMM_Std. err.", 1.0)) if status_20 == "Analyzed" else np.nan
        p_20 = float(row_20.get("GPGLMM_P>|z|", 1.0)) if status_20 == "Analyzed" else np.nan
        is_sig_20 = str(row_20.get("GPGLMM_sig", "NO")).strip().upper() == "YES" if status_20 == "Analyzed" else False

        # extract 100-tree metrics
        status_100 = "Pruned (Not Significant in 20-Tree Stage)"
        beta_100, se_100, p_100, is_sig_100 = np.nan, np.nan, np.nan, False

        if feat in df_100.index:
            row_100 = df_100.loc[feat]
            status_100 = str(row_100.get("Status", "Analyzed"))
            beta_100 = float(row_100.get("GPGLMM_Param.", 0.0))
            se_100 = float(row_100.get("GPGLMM_Std. err.", 1.0))
            p_100 = float(row_100.get("GPGLMM_P>|z|", 1.0))
            is_sig_100 = str(row_100.get("GPGLMM_sig", "NO")).strip().upper() == "YES"

        # extract baseline Verkerk et al metadata and metrics
        v_supported_coevol = "NO"
        v_bmrs_spatial = "NO"
        v_universal_text = "Unknown Proposed Universal Text Definition"
        v_universal_short = "Unknown Short Universal Definition"
        v_domain = "Unclassified"

        if feat in df_v.index:
            v_row = df_v.loc[feat]
            v_universal_text = str(v_row.get("Universal", v_universal_text))
            v_universal_short = str(v_row.get("Universal.short", v_universal_short))
            v_domain = str(v_row.get("Domain_general", v_domain))
            if str(v_row.get("supported", "NOT SIG")).strip().upper() == "SIG":
                v_supported_coevol = "YES"
            if str(v_row.get("bmrs_support", "no")).strip().lower() == "yes":
                v_bmrs_spatial = "YES"

        # define final synthesis classifications
        if is_sig_100 and v_supported_coevol == "YES":
            taxonomy = "Core Consensus Universal"
        elif is_sig_100 and v_supported_coevol == "NO":
            taxonomy = "Expanded Universal Signalling (Enhanced Isolate Random-Effects Mapping)"
        elif not is_sig_100 and is_sig_20:
            taxonomy = "Unstable Edge Case (Filtered out via 100-Tree Validation)"
        else:
            taxonomy = "Consensus Non-Significant Pair"

        master_records[feat] = {
            "Domain": v_domain,
            "Proposed_Universal_Claim": v_universal_text,
            "PU_Short": v_universal_short,
            "Verkerk_BMRS_Spatial_Stage1": v_bmrs_spatial,
            "Verkerk_Final_CoEvol": v_supported_coevol,
            "GPGLMM_20Tree_Status": status_20,
            "GPGLMM_20Tree_Beta": beta_20,
            "GPGLMM_20Tree_SE": se_20,
            "GPGLMM_20Tree_PValue": p_20,
            "GPGLMM_20Tree_IsSig": "YES" if is_sig_20 else "NO",
            "GPGLMM_100Tree_Status": status_100,
            "GPGLMM_100Tree_Beta": beta_100,
            "GPGLMM_100Tree_SE": se_100,
            "GPGLMM_100Tree_PValue": p_100,
            "GPGLMM_100Tree_IsSig": "YES" if is_sig_100 else "NO",
            "Paper_Taxonomy": taxonomy
        }

    # compile and export
    df_master = pd.DataFrame.from_dict(master_records, orient="index")

    # sort logically by taxonomy classification groups, then by statistical weight
    df_master.sort_values(by=["Paper_Taxonomy", "GPGLMM_20Tree_PValue"], ascending=[True, True], inplace=True)
    df_master.to_excel(output_master, index_label="Feature_ID")

    print("\n")
    print(" Results")
    print("==================================================================")
    print(f" Total Features Mapped                   : {len(df_master)}")
    print(f" Category [Core Consensus Universal]     : {len(df_master[df_master['Paper_Taxonomy']=='Core Consensus Universal'])}")
    print(f" Category [Expanded Universal Signalling]: {len(df_master[df_master['Paper_Taxonomy']=='Expanded Universal Signalling (Enhanced Isolate Random-Effects Mapping)'])}")
    print(f" Category [Unstable Edge Cases Filtered] : {len(df_master[df_master['Paper_Taxonomy']=='Unstable Edge Case (Filtered out via 100-Tree Validation)'])}")
    print(f" Category [Consensus Non-Significant]    : {len(df_master[df_master['Paper_Taxonomy']=='Consensus Non-Significant Pair'])}")
    print("------------------------------------------------------------------")
    print(f" Results spreadsheet saved directly to: {output_master}\n")

if __name__ == "__main__":
    generate_master_sheet()
