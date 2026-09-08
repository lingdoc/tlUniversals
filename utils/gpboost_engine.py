import os
import gzip
import re
import warnings
import pandas as pd
import numpy as np
from scipy import stats
import gpboost as gpb

# mute standard warnings to ensure clean tqdm terminal tracking progress bars
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def parse_nexus_tree_topology(trees_gz_path, num_trees=50):
    """
    Reads the zipped Nexus tree file once, maps numerical indices to Glottocodes,
    and returns a list of dictionaries mapping language tips to their parent node branch IDs.
    Gets a sample of `num_trees` (default 50).
    """
    if not os.path.exists(trees_gz_path):
        return []
    # instantiate variables
    translate_map = {}
    raw_tree_lines = []
    in_translate_block = False
    # get the tree
    with gzip.open(trees_gz_path, 'rt') as f:
        for line in f:
            line_clean = line.strip()
            line_lower = line_clean.lower()

            if line_lower.startswith("translate"):
                in_translate_block = True
                continue
            if in_translate_block and line_clean == ";":
                in_translate_block = False
                continue

            if in_translate_block and line_clean:
                parts = re.split(r'\s+', line_clean.rstrip(',').rstrip(';'))
                if len(parts) >= 2:
                    idx_token = parts[0].strip()
                    raw_label = parts[1].strip()
                    glotto_label = raw_label.split('_')[0] if '_' in raw_label else raw_label
                    translate_map[idx_token] = glotto_label
                continue

            if line_lower.startswith("tree ") or (line_clean.startswith("(") and line_clean.endswith(";")):
                raw_tree_lines.append(line_clean) # append the lines

    if not raw_tree_lines:
        return []
    # sample the num_trees from the dataset
    sampled_indices = np.linspace(0, len(raw_tree_lines) - 1, num_trees, dtype=int)
    sampled_trees_branches = []

    for idx in sampled_indices:
        line_clean = raw_tree_lines[idx]
        tree_string = line_clean.split("=", 1)[1].strip() if "=" in line_clean else line_clean
        tree_string = re.sub(r'\[.*?\]', '', tree_string)

        branch_mapping = {}
        node_counter = 10000
        stack = []
        # get the nodes
        for i, char in enumerate(tree_string):
            if char == '(':
                stack.append(node_counter)
                node_counter += 1
            elif char == ')':
                if stack: stack.pop()
            elif char != ',' and char != ';':
                match = re.match(r'^([\d]+)', tree_string[i:])
                if match:
                    token = match.group(1)
                    if token in translate_map and stack:
                        branch_mapping[translate_map[token]] = stack[-1]

        sampled_trees_branches.append(branch_mapping) # append the branches

    return sampled_trees_branches

def process_single_feature_gpboost(featfile, gldf_shared, ntrees):
    """
    Fits a Bernoulli Logit model over ntrees incorporating a continuous spatial GP matrix.
    """
    clean_path = featfile.replace("\\", "/")
    path_parts = clean_path.split("/")
    univ = path_parts[-2]
    # get the features and trees
    feature_dir = os.path.dirname(featfile)
    trees_gz_path = os.path.join(feature_dir, "pruned_tree.trees.gz")
    # dict to store stats
    stats_profile = {
        "Status": "Skipped", "Reason": "None", "Total_Languages_Found": 0,
        "Distinct_Macroareas": 0, "Distinct_Families": 0, "DV_Variance": 0.0,
        "DV_Mean": 0.0, "GPB_n_obs": 0, "GPB_Param.": np.nan, "GPB_Std. err.": np.nan,
        "GPB_z value": np.nan, "GPB_P>|z|": np.nan, "GPB_sig": "NO", "GPB_hsig": "NO", "GPB_hhsig": "NO"
    }

    try:
        if not os.path.exists(featfile):
            stats_profile["Reason"] = "Missing feature data text file"
            return univ, stats_profile
        # get the universal feature info (DV/IV are binary)
        fdf = pd.read_csv(featfile, delimiter="\t", header=None, names=["glottocode", "DV", "IV"])

        if len(fdf) < 10:
            stats_profile["Reason"] = f"Insufficient data rows (N={len(fdf)} < 10)"
            return univ, stats_profile
        # make sure glottocode is string in both dataframes
        fdf['glottocode'] = fdf['glottocode'].astype(str).str.strip().str.lower()
        gldf_shared_copy = gldf_shared.copy()
        gldf_shared_copy['glottocode'] = gldf_shared_copy['glottocode'].astype(str).str.strip().str.lower()
        # merge the universal-specific dataset with the glottolog dataset
        df = pd.merge(gldf_shared_copy, fdf, on='glottocode', how='inner')
        # keep only the following columns
        df = df.dropna(subset=['IV', 'DV', 'latitude', 'longitude', 'macroarea', 'Family_ID'])
        df['DV'] = pd.to_numeric(df['DV'], errors='coerce') # ensure the values in this column are binary
        df['IV'] = pd.to_numeric(df['IV'], errors='coerce') # ensure the values in this column are binary
        df = df.dropna(subset=['IV', 'DV']) # drop missing info
        # get some statistical info for this universal
        stats_profile["Total_Languages_Found"] = len(df)
        if len(df) > 0:
            stats_profile["Distinct_Macroareas"] = int(df['macroarea'].nunique())
            stats_profile["Distinct_Families"] = int(df['Family_ID'].nunique())
            stats_profile["DV_Variance"] = float(df['DV'].var())
            stats_profile["DV_Mean"] = float(df['DV'].mean())
        # error handling
        if len(df) < 10:
            stats_profile["Reason"] = "Data count dropped below 10 rows after data clean"
            return univ, stats_profile

        if len(np.unique(df['DV'].to_numpy())) < 2:
            stats_profile["Reason"] = "Zero variance in dependent variable"
            return univ, stats_profile

        if not os.path.exists(trees_gz_path):
            stats_profile["Reason"] = "Missing tree archive file"
            return univ, stats_profile
        # get phylogenies for this universal
        tree_branches_list = parse_nexus_tree_topology(trees_gz_path, num_trees=ntrees)
        if not tree_branches_list:
            stats_profile["Reason"] = "Tree file processing error"
            return univ, stats_profile
        # handle isolates/singletons by assigning their glottocode to maintain group integrity
        df['Family_ID'] = df['Family_ID'].fillna(df['glottocode'])
        # convert spatial and target variables to numpy arrays for GPBoost compatibility
        coords = df[['latitude', 'longitude']].to_numpy().astype(float)
        y = df['DV'].to_numpy().astype(float)
        X = df['IV'].to_numpy().astype(float)
        # add intercept term to the design matrix (X)
        X_with_intercept = np.column_stack((np.ones(len(X)), X))
        # encode grouping factors as integer codes for the mixed-effects structure
        family_factor = df['Family_ID'].astype('category').cat.codes.to_numpy()
        macro_factor = df['macroarea'].astype('category').cat.codes.to_numpy()
        # initialize result containers
        params, ses = [], []
        had_hessian_issue = False
        # iterate through each sampled phylogeny to estimate model parameters
        for branch_map in tree_branches_list:
            # map glottocodes to specific branches; fill missing with 0 (root/unassigned)
            sub_branch_series = df['glottocode'].map(branch_map).fillna(0).astype(int)
            branch_factor = sub_branch_series.astype('category').cat.codes.to_numpy()
            # construct the multi-level grouping matrix (family, macroarea, branch)
            group_data = np.column_stack((family_factor, macro_factor, branch_factor)).astype(float)

            try:
                # implement a Bernoulli Logit GPMM to capture:
                # 1. structured dependencies (via group_data)
                # 2. continuous spatial autocorrelation (via GP kernel)
                gp_model = gpb.GPModel(
                    group_data=group_data, # random effect intercepts (family, macroarea, branch)
                    gp_coords=coords, # lat/long coordinates for spatial GP component
                    cov_function="exponential", # spatial correlation decays exponentially with distance
                    likelihood="bernoulli_logit", # binary DV (presence/absence) via logit link
                    num_parallel_threads=16 # optimized for 32-core cpu with 2 workers
                )

                # configure L-BFGS optimizer
                # configure L-BFGS optimizer with accurate parameter initializations
                gp_model.set_optim_params(params={
                    "optimizer_cov": "lbfgs", # maximize marginal likelihood via L-BFGS
                    "maxit": 35, # maximum iterations allowed for convergence
                    # Order: [var_family, var_macroarea, var_branch, var_spatial, range_spatial]
                    "init_cov_pars": [0.2, 0.2, 0.2, 0.5, 1.5], 
                    "convergence_criterion": "relative_change_in_parameters",
                    "delta_rel_conv": 1e-3 # convergence threshold
                })
                # fit the model
                gp_model.fit(y=y, X=X_with_intercept)
                # get the results for this model
                coefficients = gp_model.get_coef(std_err=True, format_pandas=True)
                coef_dict = coefficients.to_dict()
                # store the results for this model
                if "Covariate_2" in coef_dict:
                    p_val = float(coef_dict["Covariate_2"].get("Param.", np.nan))
                    s_val = float(coef_dict["Covariate_2"].get("Std. err.", np.nan))

                    # live print verification
                    print(f"   [Tree Step Log] Feature: {univ} | Extracted Slope: {p_val:.4f} | Raw SE: {s_val}")

                    if np.isnan(s_val) or np.isinf(s_val) or s_val <= 0:
                        s_val = 1.0  # boundary protection fallback
                        had_hessian_issue = True

                    params.append(p_val)
                    ses.append(s_val)
            except Exception as e:
                print(f"   Execution crash on tree step: {str(e)}")
                continue
        # handle errors
        if not params:
            stats_profile["Status"] = "Failed"
            stats_profile["Reason"] = "Model did not converge on any tree configuration"
            return univ, stats_profile
        # calculate final values for the universal
        final_param = float(np.median(params))
        final_se = float(np.median(ses))

        # guard against zero-division exceptions during final reporting splits
        if final_se == 0:
            final_se = 1.0
        # calculate z and p values
        z_values = final_param / final_se
        p_values = float(2 * stats.norm.cdf(-np.abs(z_values)))

        # update final profile records
        stats_profile["Status"] = "Analyzed"
        stats_profile["Reason"] = "Hessian singular warning on sub-nodes" if had_hessian_issue else "Model completed successfully"
        stats_profile["GPB_n_obs"] = len(df)
        stats_profile["GPB_Param."] = final_param
        stats_profile["GPB_Std. err."] = final_se
        stats_profile["GPB_z value"] = z_values
        stats_profile["GPB_P>|z|"] = p_values
        # determine significance
        if p_values < 0.05:
            stats_profile["GPB_sig"] = "YES"
            if p_values < 0.01:
                stats_profile["GPB_hsig"] = "YES"
                if p_values < 0.001:
                    stats_profile["GPB_hhsig"] = "YES"
                else:
                    stats_profile["GPB_hhsig"] = "NO"
            else:
                stats_profile["GPB_hsig"] = "NO"
                stats_profile["GPB_hhsig"] = "NO"
        else:
            stats_profile["GPB_sig"] = "NO"
            stats_profile["GPB_hsig"] = "NO"
            stats_profile["GPB_hhsig"] = "NO"

        return univ, stats_profile

    except Exception as e:
        stats_profile["Status"] = "Error"
        stats_profile["Reason"] = str(e)
        return univ, stats_profile
