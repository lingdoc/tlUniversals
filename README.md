# Constraints that endure: How Galton's Problem exposes model bias in linguistic typology

This repository supports a reanalysis of the findings in [Verkerk et al (2025)](https://www.nature.com/articles/s41562-025-02325-z) with a focus on optimizing the underlying methodology for identifying significant statistical universals. Specifically, I implement a Generalized Linear Mixed Model (GLMM) with a continuous Gaussian Process (GP). I utilize the Python `gpboost` library primarily for its ability to handle Gaussian Process matrices (for spatial autocorrelation) alongside mixed effects (language family, branch, macroarea) - the Tree Boosting functionality in GPBoost is not used.

This specific architecture (GP-GLMM) allows for a more nuanced handling of language isolates than is possible with the covariance matrix used by R's `brms` library, as well as how phylogenetic evolution co-evolution is subsequently handled in `BayesTraits`. This constructive reanalysis confirms the main claims of the paper but also highlights the likely presence of a greater number of significant results than was reported.

The repository is forked from the original GitHub repo containing the data underlying the Verkerk et al 2025 paper (https://github.com/SimonGreenhill/TestingLinguisticUniversals). Datasets include (for each universal) a single coded language file and a 1000- or 100-tree sample of phylogenies. The relevant data is stored in the `tlu` folder.

To check the files and relevant statistics, run the script at `utils/check_datasets.py`.

The model is instantiated with the code in `utils/gpglmm_engine.py`.

To run the Python 20-tree model on all 191 universals, use the following script: `run_pipeline_20tree_check.py` - this produces `output/GPGLMM_01_20tree.xlsx`.

To run the 100-tree model on the 114 universals found to be significant by the 20-tree model, use the following script: `run_targeted_100tree_parallel.py` - this script is optimized for parallel processing on a 32-core cpu and produces `output/GPGLMM_02_100tree.xlsx`.

Statistical traces from the model during the 100-tree run are saved in `output/model_predictions/`.

The final unified spreadsheet (`output/Results_combined_BT_GPGLMM.xlsx`) combines the Verkerk et al results with the present analysis. To produce this file, run the script at `utils/compile_master_data_sheet.py`.

To create a plot of the significant universals and their beta coefficients, run the script at `utils/plot_universals_forest.py` - it will create a pdf in the `output` directory based on the final unified spreadsheet.

[![Forest plot](./output/universals_forest_plot.png)](./output/universals_forest_plot.pdf)
