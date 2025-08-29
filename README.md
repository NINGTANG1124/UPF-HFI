# UPF-HFI: Household Food Insecurity and Ultra-Processed Food Exposure
 
The project investigates the association between **Household Food Insecurity (HFI)** and children’s exposure to **Ultra-Processed Foods (UPF)** in the Born in Bradford (BiB) cohort.


## Analysis Process

1. **NOVA Classification** (`notebooks/NOVA_classification.ipynb`)
- Assign NOVA 1-4 categories to Intake24 dietary items
- Combine **deterministic NDNS Year 1-6 mapping** with **manual adjudication**
- Output: `outcome/NOVA_step1_outcome.xlsx`
- Manual classification based on NDNS Year 12 classification
- Output: `outcome/NOVA_step2_outcome.xlsx`

2. **Weighted UPF Percentage** (`notebooks/Weighted.ipynb`)
- Calculate the energy percentage of each child's NOVA 4 food
- Apply **weekday:weekend = 5:2 weighting** to reflect the "real week"
- Output: `outcome/weighted_upf_percent.xlsx`

3. **HFI Definition** (`notebooks/HFI.ipynb`)
- Implementing the USDA Six-Item Short Form (12-Month Reference)
- Generates raw scores (0-6), binary HFI (secure/insecure), and ordered three-level categories
- Output: `outcome/survey_with_HFI.xlsx`

4. **Modeling** (`notebooks/model.ipynb`)
- Combines weighted UPF% with HFI and covariates (age, sex, race, family size, socioeconomic status)
- Fits an **OLS regression** using HC3 robust SEs
- Modules:
- **Core A**: HFI (binary) → UPF% (adjusted for age, sex, race, number of children)
- **Interaction**: HFI × race, HFI × family size
- **Robustness**: +SES (income, employment; same population vs. new population)
- **Sensitivity**: Three-category HFI and linear trend
- Output: `outcome/merged_model.xlsx`

---

## Key Findings

- **UPF exposure** was high across all groups (approximately 70% of daily energy).
- **HFI → UPF%**: Food-insecure children consumed approximately **+2-3 percentage points** higher UPF than their food-secure peers, but confidence intervals (CIs) were wide and the estimates were not always statistically significant.
- **Race**: Asians and other groups generally had lower UPF% than Whites, but there was no strong evidence of effect modification.
- **SES**: **Adjustment tended to increase the estimated HFI effect by approximately 0.6-1.0 percentage points.
- **Sensitivity**: The three HFI categories showed a monotonic gradient (p-trend approximately 0.066).

---

## Require

-Python ≥ 3.10
- `pandas`, `numpy`, `matplotlib`, `statsmodels`, `scipy`, `openpyxl`

```bash
pip install -r requirements.txt
