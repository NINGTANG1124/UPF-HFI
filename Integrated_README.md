
# Integrated Pipeline — How to Use

This folder contains an auto-merged pipeline from your four notebooks.

## Files
- `Integrated_Pipeline.ipynb` — A master notebook with four stages:
  1. NOVA Classification
  2. Weighted Intakes
  3. HFI Variables
  4. Final Modelling
- `integrated_pipeline.py` — A linear Python script concatenating all code cells.
- `outputs/` — Figures, tables, logs, and intermediate files (created as you run).

## Suggested workflow
1. Open `Integrated_Pipeline.ipynb` and run cells top-to-bottom.
2. At the end of each stage, save any key dataframes to `outputs/intermediate/` (e.g., using `to_parquet`).
3. In the next stage, load those intermediates to keep variable names consistent.
4. Adjust any file paths or variable names that differ between the original notebooks.

*Generated on 2025-08-29 07:04.*
