import os, sys, glob
import pandas as pd
import numpy as np
from tqdm import tqdm

# force Python to check the active working folder first
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# import the engine processing routine from utils
from utils.gpglmm_engine import process_single_feature_gpglmm

if __name__ == "__main__":
    # load language data
    gldf = pd.read_csv("tlu/Glottolog_Languages.csv")
    # force the Family_ID field to stay empty if it is a missing
    gldf['Family_ID'] = gldf['Family_ID'].replace(['NaN', 'nan', 'None', ''], np.nan)
    # fill missing family fields with the language's own unique Glottocode
    # this prevents them from clumping together into a false "super-family"
    gldf['Family_ID'] = gldf['Family_ID'].fillna(gldf['glottocode']).astype(str).str.strip()
    gldf = gldf.dropna(subset=['latitude']) # remove languages without coordinates
    gldf_shared = gldf[['glottocode', 'macroarea', 'Family_ID', 'longitude', 'latitude']].copy()

    # Search directory paths
    start_directory = 'tlu/'
    file_pattern = '*data.txt'
    search_pattern = os.path.join(start_directory, '**', file_pattern)
    matching_files = [x for x in glob.glob(search_pattern, recursive=True) if "_summary" not in x]

    output_excel = "output/GPGLMM_01_20tree.xlsx"

    # set number of phylogenies to 20 for identifying initial significant correlations
    TARGET_NTREES = 20

    print(" Launching Sequential Python-GPGLMM Spatio-Phylogenetic Pipeline...")
    print(f" Tree Sampling Depth: Processing {TARGET_NTREES} posterior tree variations per feature.")
    print(f" Total Linguistic Features queued: {len(matching_files)}\n")

    xdict = {}

    if os.path.isfile(output_excel):
        response = input(f"The file '{os.path.basename(output_excel)}' already exists.\nContinuing will overwrite the file.\nDo you wish to proceed? (y/n): ")
        # Safely default to canceling unless the user explicitly confirms
        if response.lower() not in ['y', 'yes']:
            print("Operation canceled. File was not overwritten.")
            exit()
        else:
            # standard loop structure for full-thread utility
            for ffile in tqdm(matching_files, desc="Running Python GPGLMM Engine", unit="feature"):
                univ_key, stats_row = process_single_feature_gpglmm(ffile, gldf_shared, TARGET_NTREES)
                xdict[univ_key] = stats_row

    # compile dataset output rows and export summary workbook data
    if xdict:
        df_final = pd.DataFrame.from_dict(xdict, orient='index')
        ordered_cols = [
            "Status", "Reason", "Total_Languages_Found", "Distinct_Macroareas",
            "Distinct_Families", "DV_Variance", "DV_Mean", "GPGLMM_n_obs",
            "GPGLMM_Param.", "GPGLMM_Std. err.", "GPGLMM_z value", "GPGLMM_P>|z|",
            "GPGLMM_sig"
        ]
        df_final = df_final[ordered_cols]
        df_final.to_excel(output_excel, index_label="Feature_ID")

        print(f"\n Complete! Results saved directly to {output_excel}")

        print("\n=====================================")
        print("LINGUISTIC UNIVERSALS SUMMARY REPORT")
        print("=====================================")
        total_features = len(df_final)
        analyzed_count = (df_final['Status'] == "Analyzed").sum()
        skipped_count  = (df_final['Status'] == "Skipped").sum()
        failed_count   = (df_final['Status'] == "Failed").sum()

        sig_mask = (df_final['GPGLMM_sig'] == "YES") & (df_final['Status'] == "Analyzed")
        total_significant = sig_mask.sum()
        pos_sig = ((df_final['GPGLMM_Param.'] > 0) & sig_mask).sum()
        neg_sig = ((df_final['GPGLMM_Param.'] < 0) & sig_mask).sum()

        print(f"🔹 Total Typological Features Processed : {total_features}")
        print(f"🔹 Total Natively Analyzed Models       : {analyzed_count}")
        print(f"🔹 Total Skipped Features               : {skipped_count}")
        print(f"🔹 Total Regression Convergence Failures: {failed_count}")
        print("--------------------------------------------------")
        if analyzed_count > 0:
            print(f"🔹 Significant Results (p < 0.05)       : {total_significant} ({total_significant/analyzed_count:.1%} of analyzed models)")
            print(f"   ▶️ Positive Significant (+)          : {pos_sig}")
            print(f"   ▶️ Negative Significant (-)          : {neg_sig}")
        print("==================================================\n")
