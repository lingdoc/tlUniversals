import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from scipy import stats

def generate_supplementary_tables(df_master):
    """
    Generates two separate Excel files:
    1. Table_A_Consensus.xlsx (Verkerk significant)
    2. Table_B_Expansion.xlsx (GPGLMM significant, Verkerk not)
    Includes GPGLMM Beta values in both.
    """

    # --- CONFIGURATION (Ensure these match your master dataframe exactly) ---
    VERKERK_SIG_COL = 'Verkerk_Final_CoEvol'
    GPGLMM_SIG_COL = 'GPGLMM_100Tree_IsSig'
    GPGLMM_BETA_COL = 'GPGLMM_100Tree_Beta'
    SHORT_NAME_COL = 'PU_Short'
    DEF_COL = 'Proposed_Universal_Claim'
    CODE_COL = 'Feature_ID'
    # -------------------------------------------------------------------------

    print(" Processing Master Dataset...")

    # prep dataframe
    df = df_master.copy()
    if df.index.name != CODE_COL:
        df = df.reset_index()
    else:
        df = df.reset_index()

    # helper to handle significance ("YES" strings and numeric values)
    def is_sig(val):
        if pd.isna(val): return False
        if isinstance(val, str): return val.strip().upper() == "YES"
        return val > 0

    # apply masking
    verkerk_sig_mask = df[VERKERK_SIG_COL].apply(is_sig)
    gpglmm_sig_mask = df[GPGLMM_SIG_COL].apply(is_sig)

    # generate A (consensus / Verkerk supported)
    # filter: Verkerk is significant
    table_a_rows = df[verkerk_sig_mask].copy()

    def get_source_a(row):
        # check if GPGLMM also agrees
        if is_sig(row[GPGLMM_SIG_COL]):
            return "Both"
        return "Verkerk"

    table_a_rows['Support_Source'] = table_a_rows.apply(get_source_a, axis=1)

    # select relevant columns including Beta
    table_a = table_a_rows[[CODE_COL, SHORT_NAME_COL, DEF_COL, GPGLMM_BETA_COL, 'Support_Source']].copy()
    table_a.columns = ['Code', 'Short_Name', 'Definition', 'GPGLMM_Beta', 'Support_Source']

    # generate table B (expansion / GPGLMM supported)
    # filter: GPGLMM is significant AND Verkerk is NOT significant
    table_b_rows = df[gpglmm_sig_mask & ~verkerk_sig_mask].copy()
    table_b_rows['Support_Source'] = "GPGLMM"

    table_b = table_b_rows[[CODE_COL, SHORT_NAME_COL, DEF_COL, GPGLMM_BETA_COL, 'Support_Source']].copy()
    table_b.columns = ['Code', 'Short_Name', 'Definition', 'GPGLMM_Beta', 'Support_Source']

    # generate table C to conduct binomial test on 109 significant features
    table_c_rows = df[gpglmm_sig_mask].copy()
    # column containing the Beta coefficients.
    target_column = 'GPGLMM_100Tree_Beta'
    # count how many betas are greater than zero
    successes = (table_c_rows[target_column] > 0).sum()
    total_n = len(table_c_rows)
    failures = (table_c_rows[target_column] < 0).sum()

    print(f"--- Analysis Summary ---")
    print(f"Total Significant Features Analyzed: {total_n}")
    print(f"Positive Betas (Successes): {successes}")
    print(f"Negative Betas (Failures): {failures}")

    # Binomial Test (use 'greater' to test if there is a significant bias toward POSITIVE coefficients)
    p_value = stats.binomtest(successes, n=total_n, p=0.5, alternative='greater').pvalue

    print(f"\n--- Statistical Result ---")
    print(f"P-value: {p_value:.2e}")

    if p_value < 0.05:
        print("Conclusion: Reject H0. There is a significant directional bias in the coefficients.")
    else:
        print("Conclusion: Fail to reject H0. The distribution of signs is consistent with randomness.")

    # export tables to excel
    try:
        table_a.to_excel("../output/Table_A_Consensus.xlsx", index=False)
        table_b.to_excel("../output/Table_B_Expansion.xlsx", index=False)

        print(f" Success!")
        print(f" Saved: ../output/Table_A_Consensus.xlsx ({len(table_a)} rows)")
        print(f" Saved: ../output/Table_B_Expansion.xlsx ({len(table_b)} rows)")

    except Exception as e:
        print(f" Error saving Excel files: {e}")

    return table_a, table_b


def generate_proportional_quadrant_plot():
    master_file = "../output/Results_combined_BT_GPGLMM.xlsx"
    output_pdf = "../output/universals_forest_plot.pdf"

    df_master = pd.read_excel(master_file, index_col="Feature_ID")

    # generate the tables
    generate_supplementary_tables(df_master)

    df_sig = df_master[df_master["GPGLMM_100Tree_IsSig"] == "YES"].copy()
    df_sig["abs_beta"] = df_sig["GPGLMM_100Tree_Beta"].abs()
    df_sig["Domain"] = df_sig["Domain"].astype(str).str.strip().str.lower()

    categories = ["broad word order", "narrow word order", "hierarchy", "other"]
    counts = [len(df_sig[df_sig["Domain"] == cat]) for cat in categories]

    plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9})
    fig = plt.figure(figsize=(8.27, 11.69))

    gs_left_master = gridspec.GridSpec(1, 1, left=0.22, right=0.48, top=0.88, bottom=0.07)
    gs_right_master = gridspec.GridSpec(1, 1, left=0.55, right=0.76, top=0.88, bottom=0.07)

    gs_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_left_master[0, 0], height_ratios=[counts[0], counts[2]], hspace=0.25)
    gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_right_master[0, 0], height_ratios=[counts[1], counts[3]], hspace=0.25)

    layout_mapping = [(gs_left, 0, False), (gs_right, 0, True), (gs_left, 1, False), (gs_right, 1, True)]
    active_axes = []

    for cat, count, (gs_target, row_idx, flip_y_axis) in zip(categories, counts, layout_mapping):
        ax = fig.add_subplot(gs_target[row_idx])
        active_axes.append((ax, cat, count))
        df_sub = df_sig[df_sig["Domain"] == cat].copy()
        df_sub.sort_values(by="abs_beta", ascending=True, inplace=True)
        y_pos = range(len(df_sub))
        colors = ['#1f4e79' if r["Verkerk_Final_CoEvol"] == "YES" else '#5b9bd5' for _, r in df_sub.iterrows()]
        min_whisker, max_whisker = 0.0, 0.0
        for i, (_, row) in enumerate(df_sub.iterrows()):
            b, se = row["GPGLMM_100Tree_Beta"], row["GPGLMM_100Tree_SE"]
            ci_low, ci_high = b - (1.96 * se), b + (1.96 * se)
            if ci_low < min_whisker: min_whisker = ci_low
            if ci_high > max_whisker: max_whisker = ci_high
            ax.plot([ci_low, ci_high], [i, i], color='#7f7f7f', linewidth=0.9, zorder=1)
        ax.scatter(df_sub["GPGLMM_100Tree_Beta"], y_pos, c=colors, s=14, edgecolor='black', linewidth=0.4, zorder=2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_sub["PU_Short"], fontsize=6)
        if flip_y_axis:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.6, alpha=0.6)
        ax.grid(True, linestyle=':', alpha=0.35)
        x_pad = (max_whisker - min_whisker) * 0.10 if max_whisker != min_whisker else 1.0
        ax.set_xlim(min_whisker - x_pad, max_whisker + x_pad)
        if count > 0: ax.set_ylim(-0.75, count - 0.25)

    fig.text(0.5, 0.96, "Distribution of Validated Spatio-Phylogenetic Universals", fontsize=12, fontweight='bold', ha='center', va='top')
    legend_elements = [
        Line2D([], [], marker='o', color='w', markerfacecolor='#1f4e79', markeredgecolor='black', markersize=6, label='Core Consensus Universal'),
        Line2D([], [], marker='o', color='w', markerfacecolor='#5b9bd5', markeredgecolor='black', markersize=6, label='Isolate Power Expansion')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.93), ncol=2, frameon=False, fontsize=8)
    fig.text(0.49, 0.03, "Estimated Fixed-Effect Slope Parameter (Beta Coefficient)", fontsize=9, fontweight='bold', ha='center', va='bottom')
    plt.gcf().canvas.draw()

    for ax, cat, count in active_axes:
        bbox = ax.get_position()
        clean_title = cat.title() if cat != 'other' else 'Other Morphosyntactic'
        fig.text(bbox.x0, bbox.y1 + 0.012, f"Category: {clean_title} (n={count})", fontsize=9, fontweight='bold', ha='left', va='bottom')

    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', pad_inches=0.02) # export pdf
    plt.savefig(output_pdf[:-3]+"png", dpi=300, bbox_inches='tight', pad_inches=0.02) # export png
    print(f" Success! Forest plot saved to '{output_pdf}'")

if __name__ == "__main__":
    generate_proportional_quadrant_plot()
