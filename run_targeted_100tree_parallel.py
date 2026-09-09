import os, sys, glob
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# optimize for ryzen 9 7950X cores
os.environ["OMP_NUM_THREADS"] = "16"       # lock each active process to 16 threads
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["VECLIB_MAXIMUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"

# force Python to check active working folder first
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
# import the engine from utils
from utils.gpboost_engine import process_single_feature_gpboost

if __name__ == "__main__":
    master_results_file = "output/GPBoost_01_20tree.xlsx"
    output_excel = "output/GPBoost_02_100tree.xlsx"

    if not os.path.exists(master_results_file):
        print(f" Error: Cannot find baseline file '{master_results_file}'")
        exit()

    # load language data and coordinates
    gldf = pd.read_csv("tlu/Glottolog_Languages.csv")
    # ensure empty family ids are empty
    gldf['Family_ID'] = gldf['Family_ID'].replace(['NaN', 'nan', 'None', ''], pd.NA)
    # replace empty rows with glottocode
    gldf['Family_ID'] = gldf['Family_ID'].fillna(gldf['glottocode']).astype(str).str.strip()
    gldf = gldf.dropna(subset=['latitude']) # drop rows with no coordinates
    gldf_shared = gldf[['glottocode', 'macroarea', 'Family_ID', 'longitude', 'latitude']].copy()

    # read the 114 significant Feature IDs directly from the previous spreadsheet (20-tree run)
    df_baseline = pd.read_excel(master_results_file)
    df_baseline.rename(columns={df_baseline.columns[0]: "Feature_ID"}, inplace=True)
    sig_features = df_baseline[df_baseline["GPB_sig"] == "YES"]["Feature_ID"].astype(str).tolist()

    # map the targeted IDs to their actual data text file paths
    start_directory = 'tlu/'
    files_to_process = []
    for feat_id in sig_features:
        match_pattern = os.path.join(start_directory, f"*{feat_id}*", "*data.txt")
        matched_files = glob.glob(match_pattern)
        if matched_files:
            files_to_process.append(matched_files[0])

    # target the 100-tree depth
    TARGET_NTREES = 100

    # set max workers
    MAX_WORKERS = 2

    print(" Launching High-Speed Parallel 100-Tree Validation Pipeline...")
    print(f" Server Capacity: Utilizing {MAX_WORKERS} parallel feature workers simultaneously.")
    print(f" Targeted Features Queue Depth : {len(files_to_process)} Universals")
    print(f" Tree Sampling Depth          : Evaluating ALL {TARGET_NTREES} variations per universal.\n")

    xdict = {}

    run = True
    if os.path.isfile(output_excel):
        response = input(f"The file '{os.path.basename(output_excel)}' already exists.\nContinuing will overwrite the file.\nDo you wish to proceed? (y/n): ")
        # Safely default to canceling unless the user explicitly confirms
        if response.lower() not in ['y', 'yes']:
            print("Operation canceled. File was not overwritten.")
            run = False
            exit()
        else:
            run = True

    if run:
        # asynchronous process worker submission layer
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file = {
                executor.submit(process_single_feature_gpboost, ffile, gldf_shared, TARGET_NTREES): ffile
                for ffile in files_to_process
            }

            for future in tqdm(as_completed(future_to_file), total=len(files_to_process), desc="Running 100-Tree Engine", unit="feature"):
                univ_key, stats_row = future.result()
                xdict[univ_key] = stats_row

    # compile dataset output rows and export workbook data
    if xdict:
        df_final = pd.DataFrame.from_dict(xdict, orient='index')
        ordered_cols = [
            "Status", "Reason", "Total_Languages_Found", "Distinct_Macroareas",
            "Distinct_Families", "DV_Variance", "DV_Mean", "GPB_n_obs",
            "GPB_Param.", "GPB_Std. err.", "GPB_z value", "GPB_P>|z|",
            "GPB_sig"
        ]
        df_final = df_final[ordered_cols]
        df_final.to_excel(output_excel, index_label="Feature_ID")

        print(f"\n Parallel Verification Complete! Data saved directly to {output_excel}")
