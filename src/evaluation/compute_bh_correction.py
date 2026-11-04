import json
import numpy as np
from scipy import stats

def run_bh_audit():
    json_path = "c:/Thesis/reports/final/final_statistical_analysis.json"
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    conditions = list(data.keys())
    raw_p_values = [data[c]["p_value"] for c in conditions]
    
    print(f"Loaded {len(raw_p_values)} p-values from {json_path}")
    assert len(raw_p_values) == 11, f"Expected 11 conditions, got {len(raw_p_values)}"

    # Apply Benjamini-Hochberg correction using Scipy
    bh_p_values = stats.false_discovery_control(raw_p_values, method="bh")

    print("\n=== BENJAMINI-HOCHBERG FDR CORRECTION RESULTS ===")
    print(f"{'Condition':<26} | {'Original p-value':<22} | {'BH-Adjusted p-value':<22} | {'Significant (q < 0.05)?':<22}")
    print("-" * 100)

    sig_count = 0
    max_bh_p = -1.0

    for i, c in enumerate(conditions):
        orig_p = raw_p_values[i]
        bh_p = bh_p_values[i]
        is_sig = bh_p < 0.05
        if is_sig:
            sig_count += 1
        if bh_p > max_bh_p:
            max_bh_p = bh_p

        print(f"{c:<26} | {orig_p:<22.10e} | {bh_p:<22.10e} | {str(is_sig):<22}")

    print("\n=== SUMMARY STATISTICS ===")
    print(f"1. Total comparisons evaluated:             {len(bh_p_values)}")
    print(f"2. Number of significant comparisons (q<0.05): {sig_count} / {len(bh_p_values)}")
    print(f"3. Largest BH-adjusted p-value:              {max_bh_p:.10e}")

if __name__ == "__main__":
    run_bh_audit()
