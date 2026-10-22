# %% 
# Smart Region Matcher with LLM-based fuzzy matching

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import os
from pathlib import Path
from difflib import SequenceMatcher
import re

def validate_dataframe(df, name, required_columns):
    """Validate dataframe has required columns and non-empty."""
    if df.empty:
        raise ValueError(f"{name} dataframe is empty")
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{name} dataframe missing required columns: {missing_cols}")
    return True

def clean_region_name(name):
    """Clean region name for better matching."""
    # Remove extra spaces and standardize
    name = re.sub(r'\s+', ' ', name.strip())
    # Standardize common abbreviations
    name = name.replace('_', '-')
    return name

def calculate_similarity(name1, name2):
    """Calculate similarity between two region names."""
    # Clean both names
    clean1 = clean_region_name(name1.lower())
    clean2 = clean_region_name(name2.lower())
    
    # Exact match
    if clean1 == clean2:
        return 1.0
    
    # Sequence similarity
    seq_sim = SequenceMatcher(None, clean1, clean2).ratio()
    
    # Word-level similarity
    words1 = set(clean1.split('-'))
    words2 = set(clean2.split('-'))
    if words1 and words2:
        word_sim = len(words1.intersection(words2)) / len(words1.union(words2))
    else:
        word_sim = 0
    
    # Combined similarity (weighted)
    combined_sim = 0.6 * seq_sim + 0.4 * word_sim
    
    return combined_sim

def smart_region_matcher(llm_regions, atlas_regions, similarity_threshold=0.7):
    """
    Intelligently match region names using similarity scoring.
    """
    print("🧠 SMART REGION MATCHING")
    print("="*50)
    
    matches = {}
    unmatched_llm = []
    unmatched_atlas = []
    
    # Create sets for tracking
    matched_atlas = set()
    
    # For each LLM region, find best atlas match
    for llm_region, llm_hem in llm_regions:
        best_match = None
        best_similarity = 0
        
        for atlas_region, atlas_hem in atlas_regions:
            # Only match within same hemisphere
            if llm_hem != atlas_hem:
                continue
                
            # Skip if already matched
            if (atlas_region, atlas_hem) in matched_atlas:
                continue
            
            similarity = calculate_similarity(llm_region, atlas_region)
            
            if similarity > best_similarity and similarity >= similarity_threshold:
                best_similarity = similarity
                best_match = (atlas_region, atlas_hem)
        
        if best_match:
            matches[(llm_region, llm_hem)] = best_match
            matched_atlas.add(best_match)
            print(f"✅ {llm_region} ({llm_hem}) → {best_match[0]} ({best_match[1]}) [sim: {best_similarity:.3f}]")
        else:
            unmatched_llm.append((llm_region, llm_hem))
            print(f"❌ {llm_region} ({llm_hem}) → No match found")
    
    # Find unmatched atlas regions
    for atlas_region, atlas_hem in atlas_regions:
        if (atlas_region, atlas_hem) not in matched_atlas:
            unmatched_atlas.append((atlas_region, atlas_hem))
    
    print(f"\n📊 MATCHING SUMMARY:")
    print(f"  Successful matches: {len(matches)}")
    print(f"  Unmatched LLM regions: {len(unmatched_llm)}")
    print(f"  Unmatched Atlas regions: {len(unmatched_atlas)}")
    
    return matches, unmatched_llm, unmatched_atlas

def apply_region_mapping(df, mapping, is_llm_data=True):
    """Apply region mapping to dataframe."""
    df_mapped = df.copy()
    
    if is_llm_data:
        # For LLM data, map from LLM names to Atlas names
        region_map = {}
        for (llm_region, llm_hem), (atlas_region, atlas_hem) in mapping.items():
            region_map[(llm_region, llm_hem)] = (atlas_region, atlas_hem)
        
        # Apply mapping
        for i, row in df_mapped.iterrows():
            key = (row['region'], row['hemisphere'])
            if key in region_map:
                new_region, new_hem = region_map[key]
                df_mapped.at[i, 'region'] = new_region
                df_mapped.at[i, 'hemisphere'] = new_hem
    
    return df_mapped

def manual_region_fixes():
    """
    Manual fixes for common region name variations.
    Add known mappings here.
    """
    manual_mapping = {
        # Format: (llm_region, hemisphere): (atlas_region, hemisphere)
        # Example mappings - you can add more based on your specific mismatches
        ('G_temp_sup-Plan_polar', 'Left'): ('G_temp_sup-Plan_polar', 'Left'),
        ('G_temp_sup-Plan_polar', 'Right'): ('G_temp_sup-Plan_polar', 'Right'),
        ('G_temp_sup-Lateral', 'Left'): ('G_temp_sup-Lateral', 'Left'),
        ('G_temp_sup-Lateral', 'Right'): ('G_temp_sup-Lateral', 'Right'),
        # Add more mappings as needed
    }
    return manual_mapping

try:
    # 1. Load and validate CSV files
    required_files = ['synthetic_data/llm_predictions.csv', 'synthetic_data/synthetic_data.csv']
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Required file not found: {file}")
    
    llm_df = pd.read_csv('synthetic_data/llm_predictions.csv')
    patient_df = pd.read_csv('synthetic_data/synthetic_data.csv')
    
    # Validate dataframes
    required_columns = ['region', 'hemisphere', 'rank', 'z_score_atrophy']
    validate_dataframe(llm_df, 'LLM predictions', required_columns)
    validate_dataframe(patient_df, 'Patient data', required_columns)
    
    print(f"📊 INITIAL DATA LOADED:")
    print(f"  LLM predictions: {len(llm_df)} rows")
    print(f"  Patient data: {len(patient_df)} rows")
    
    # 2. Get unique regions from each dataset
    llm_regions = set(zip(llm_df['region'], llm_df['hemisphere']))
    atlas_regions = set(zip(patient_df['region'], patient_df['hemisphere']))
    
    print(f"\n🔍 REGION ANALYSIS:")
    print(f"  Unique LLM regions: {len(llm_regions)}")
    print(f"  Unique Atlas regions: {len(atlas_regions)}")
    
    # 3. Try exact matching first
    exact_matches = llm_regions.intersection(atlas_regions)
    unmatched_llm = llm_regions - atlas_regions
    unmatched_atlas = atlas_regions - llm_regions
    
    print(f"\n✅ Exact matches: {len(exact_matches)}")
    print(f"❌ Unmatched LLM regions: {len(unmatched_llm)}")
    print(f"❌ Unmatched Atlas regions: {len(unmatched_atlas)}")
    
    # 4. Apply smart matching for unmatched regions
    if unmatched_llm:
        print(f"\n🧠 APPLYING SMART MATCHING FOR {len(unmatched_llm)} UNMATCHED REGIONS:")
        smart_matches, still_unmatched_llm, still_unmatched_atlas = smart_region_matcher(
            unmatched_llm, unmatched_atlas, similarity_threshold=0.6
        )
        
        # 5. Apply manual fixes if needed
        manual_fixes = manual_region_fixes()
        print(f"\n🔧 APPLYING MANUAL FIXES:")
        final_mapping = {}
        final_mapping.update({k: k for k in exact_matches})  # Keep exact matches
        final_mapping.update(smart_matches)  # Add smart matches
        final_mapping.update(manual_fixes)  # Add manual fixes
        
        print(f"  Total mappings: {len(final_mapping)}")
        
        # 6. Apply mapping to LLM data
        llm_df_mapped = apply_region_mapping(llm_df, final_mapping, is_llm_data=True)
        
        # 7. Now try merging again
        merged = pd.merge(
            llm_df_mapped,
            patient_df,
            on=['region', 'hemisphere'],
            suffixes=('_llm', '_patient')
        )
        
        print(f"\n🎯 FINAL MERGE RESULTS:")
        print(f"  Successfully matched: {len(merged)} region pairs")
        
        if len(merged) == 0:
            print("❌ Still no matches found after smart matching!")
            print("You may need to add more manual fixes.")
            
            # Show what still doesn't match
            remaining_llm = set(zip(llm_df_mapped['region'], llm_df_mapped['hemisphere']))
            remaining_atlas = set(zip(patient_df['region'], patient_df['hemisphere']))
            still_no_match = remaining_llm - remaining_atlas
            
            print(f"\nRegions still not matching:")
            for region, hem in sorted(still_no_match):
                print(f"  {region} ({hem})")
            
            raise ValueError("Insufficient matches after smart matching")
    
    else:
        # All regions matched exactly
        merged = pd.merge(
            llm_df,
            patient_df,
            on=['region', 'hemisphere'],
            suffixes=('_llm', '_patient')
        )
    
    # 8. Continue with analysis (same as before)
    x_llm = merged['rank_llm'].values.reshape(-1, 1)
    y_llm = merged['z_score_atrophy_llm'].values
    x_patient = merged['rank_patient'].values.reshape(-1, 1)
    y_patient = merged['z_score_atrophy_patient'].values
    
    # Validate arrays
    if len(x_llm) < 2:
        raise ValueError("Insufficient data points for regression analysis")
    
    # Run linear regression
    reg = LinearRegression().fit(x_llm, y_patient)
    y_pred = reg.predict(x_llm)
    r2 = reg.score(x_llm, y_patient)
    rmse = np.sqrt(mean_squared_error(y_patient, y_pred))
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Z-score correlation
    plt.subplot(2, 3, 1)
    plt.scatter(y_llm, y_patient, alpha=0.7, color='blue', s=50)
    min_val = min(min(y_llm), min(y_patient))
    max_val = max(max(y_llm), max(y_patient))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect correlation')
    plt.xlabel("LLM Predicted Z-Score")
    plt.ylabel("Patient Actual Z-Score")
    plt.title("Z-Score Correlation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Rank correlation
    plt.subplot(2, 3, 2)
    plt.scatter(merged['rank_llm'], merged['rank_patient'], alpha=0.7, color='green', s=50)
    plt.xlabel("LLM Predicted Rank")
    plt.ylabel("Patient Actual Rank")
    plt.title("Rank Correlation")
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Parametric curves
    plt.subplot(2, 3, 3)
    plt.plot(x_llm.flatten(), y_llm, 'o-', label='LLM Predictions', color='blue', alpha=0.7, markersize=6)
    plt.plot(x_patient.flatten(), y_patient, 's-', label='Patient Data', color='red', alpha=0.7, markersize=6)
    plt.xlabel("Rank")
    plt.ylabel("Z-Score Atrophy")
    plt.title("Parametric Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Residuals
    plt.subplot(2, 3, 4)
    residuals = y_patient - y_pred
    plt.scatter(y_pred, residuals, alpha=0.7, color='purple', s=50)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Regression line
    plt.subplot(2, 3, 5)
    plt.scatter(x_llm.flatten(), y_patient, alpha=0.7, color='blue', s=50)
    plt.plot(x_llm.flatten(), y_pred, 'r-', alpha=0.8, linewidth=2, label=f'Regression Line (R²={r2:.3f})')
    plt.xlabel("LLM Predicted Rank")
    plt.ylabel("Patient Actual Z-Score")
    plt.title("Regression Analysis")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Error distribution
    plt.subplot(2, 3, 6)
    plt.hist(residuals, bins=10, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title(f"Error Distribution (RMSE={rmse:.3f})")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print comprehensive results
    print("\n" + "="*80)
    print("📈 COMPREHENSIVE ANALYSIS RESULTS")
    print("="*80)
    
    # Calculate correlations
    z_score_corr = np.corrcoef(y_llm, y_patient)[0,1]
    rank_corr = np.corrcoef(merged['rank_llm'], merged['rank_patient'])[0,1]
    
    print(f"📊 CORRELATION ANALYSIS:")
    print(f"  Z-Score Correlation: {z_score_corr:.4f}")
    print(f"  Rank Correlation:    {rank_corr:.4f}")
    
    print(f"\n📈 REGRESSION ANALYSIS:")
    print(f"  Model: Patient_Z_Score = {reg.coef_[0]:.4f} × LLM_Rank + {reg.intercept_:.4f}")
    print(f"  R² Score:    {r2:.4f}")
    print(f"  RMSE:        {rmse:.4f}")
    
    print(f"\n✅ SUCCESSFULLY MATCHED REGIONS ({len(merged)} total):")
    display_df = merged[['region', 'hemisphere', 'rank_llm', 'z_score_atrophy_llm', 'rank_patient', 'z_score_atrophy_patient']].copy()
    display_df.columns = ['Region', 'Hemisphere', 'LLM_Rank', 'LLM_Z_Score', 'Patient_Rank', 'Patient_Z_Score']
    display_df = display_df.sort_values(['LLM_Rank'])
    print(display_df.to_string(index=False))
    
    # Performance interpretation
    print(f"\n🎯 PERFORMANCE INTERPRETATION:")
    if z_score_corr > 0.7:
        print("  🟢 EXCELLENT: Strong correlation between LLM and actual z-scores")
    elif z_score_corr > 0.5:
        print("  🟡 GOOD: Moderate correlation between LLM and actual z-scores")
    elif z_score_corr > 0.3:
        print("  🟠 FAIR: Weak correlation between LLM and actual z-scores")
    else:
        print("  🔴 POOR: Very weak correlation between LLM and actual z-scores")
    
    if rank_corr > 0.7:
        print("  🟢 EXCELLENT: Strong rank correlation - LLM ordering matches patient data well")
    elif rank_corr > 0.5:
        print("  🟡 GOOD: Moderate rank correlation - LLM ordering somewhat matches patient data")
    elif rank_corr > 0.3:
        print("  🟠 FAIR: Weak rank correlation - LLM ordering poorly matches patient data")
    else:
        print("  🔴 POOR: Very weak rank correlation - LLM ordering does not match patient data")

except FileNotFoundError as e:
    print(f"❌ File Error: {str(e)}")
except ValueError as e:
    print(f"❌ Data Error: {str(e)}")
except Exception as e:
    print(f"❌ Unexpected Error: {str(e)}")
    import traceback
    traceback.print_exc()

# %%

# Enhanced Smart Region Matcher with Advanced Statistical Analysis & Better Visualizations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr, ttest_rel
import statsmodels.api as sm
import os
import re
from difflib import SequenceMatcher

sns.set(style="whitegrid", palette="muted", font_scale=1.1)

def clean_region_name(name):
    name = re.sub(r'\s+', ' ', name.strip())
    name = name.replace('_', '-')
    return name

def calculate_similarity(name1, name2):
    name1, name2 = clean_region_name(name1.lower()), clean_region_name(name2.lower())
    if name1 == name2:
        return 1.0
    seq_sim = SequenceMatcher(None, name1, name2).ratio()
    words1, words2 = set(name1.split('-')), set(name2.split('-'))
    word_sim = len(words1 & words2) / len(words1 | words2) if words1 and words2 else 0
    return 0.6 * seq_sim + 0.4 * word_sim

def smart_match(llm_set, atlas_set, threshold=0.7):
    matches, unmatched_llm, matched_atlas = {}, [], set()
    for llm in llm_set:
        best_match, best_score = None, 0
        for atlas in atlas_set:
            if llm[1] != atlas[1] or atlas in matched_atlas:
                continue
            score = calculate_similarity(llm[0], atlas[0])
            if score > best_score and score >= threshold:
                best_match, best_score = atlas, score
        if best_match:
            matches[llm] = best_match
            matched_atlas.add(best_match)
        else:
            unmatched_llm.append(llm)
    unmatched_atlas = atlas_set - matched_atlas
    return matches, unmatched_llm, unmatched_atlas

def apply_mapping(df, mapping):
    df = df.copy()
    for idx, row in df.iterrows():
        key = (row['region'], row['hemisphere'])
        if key in mapping:
            df.at[idx, 'region'], df.at[idx, 'hemisphere'] = mapping[key]
    return df

def run_regression_analysis(merged):
    x = merged['rank_llm'].values.reshape(-1, 1)
    y = merged['z_score_atrophy_patient'].values
    reg = LinearRegression().fit(x, y)
    pred = reg.predict(x)
    r2 = reg.score(x, y)
    rmse = np.sqrt(mean_squared_error(y, pred))
    return reg, pred, r2, rmse

def run_stats(y1, y2):
    return {
        'pearson': pearsonr(y1, y2),
        'spearman': spearmanr(y1, y2),
        'ttest': ttest_rel(y1, y2)
    }

def plot_results(merged, reg, pred, r2, rmse):
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    sns.scatterplot(x='z_score_atrophy_llm', y='z_score_atrophy_patient', data=merged, ax=axs[0, 0])
    axs[0, 0].plot([merged['z_score_atrophy_llm'].min(), merged['z_score_atrophy_llm'].max()],
                  [merged['z_score_atrophy_llm'].min(), merged['z_score_atrophy_llm'].max()],
                  'r--', alpha=0.6)
    axs[0, 0].set_title("Z-Score Correlation")

    sns.scatterplot(x='rank_llm', y='rank_patient', data=merged, ax=axs[0, 1])
    axs[0, 1].set_title("Rank Correlation")

    sns.lineplot(x='rank_llm', y='z_score_atrophy_llm', data=merged, ax=axs[0, 2], label='LLM')
    sns.lineplot(x='rank_patient', y='z_score_atrophy_patient', data=merged, ax=axs[0, 2], label='Patient')
    axs[0, 2].set_title("Parametric Comparison")

    residuals = merged['z_score_atrophy_patient'] - pred
    sns.residplot(x=pred, y=residuals, lowess=True, ax=axs[1, 0], line_kws={'color': 'red'})
    axs[1, 0].set_title("Residuals vs Prediction")

    sns.regplot(x='rank_llm', y='z_score_atrophy_patient', data=merged, ax=axs[1, 1], line_kws={'label': f"R²={r2:.3f}"})
    axs[1, 1].legend()
    axs[1, 1].set_title("Linear Regression")

    sns.histplot(residuals, kde=True, ax=axs[1, 2])
    axs[1, 2].set_title(f"Residual Distribution (RMSE={rmse:.3f})")

    plt.tight_layout()
    plt.show()

def bland_altman_plot(y1, y2):
    mean = np.mean([y1, y2], axis=0)
    diff = y1 - y2
    md = np.mean(diff)
    sd = np.std(diff)

    plt.figure(figsize=(7, 5))
    plt.scatter(mean, diff, alpha=0.6)
    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='red', linestyle='--')
    plt.axhline(md - 1.96*sd, color='red', linestyle='--')
    plt.title('Bland-Altman Plot')
    plt.xlabel('Mean Z-Score')
    plt.ylabel('Difference in Z-Scores')
    plt.grid(True)
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Extract variables for analysis
x_llm = merged['rank_llm'].values.reshape(-1, 1)
y_llm = merged['z_score_atrophy_llm'].values
y_patient = merged['z_score_atrophy_patient'].values

# Linear regression: predict patient z-score from LLM rank
reg = LinearRegression().fit(x_llm, y_patient)
y_pred = reg.predict(x_llm)
r2 = reg.score(x_llm, y_patient)
rmse = np.sqrt(mean_squared_error(y_patient, y_pred))

# Plotting
plt.figure(figsize=(15, 10))

# Z-score correlation plot
plt.subplot(2, 3, 1)
plt.scatter(y_llm, y_patient, color='blue', alpha=0.7, s=50)
plt.plot([min(y_llm), max(y_llm)], [min(y_llm), max(y_llm)], 'r--', label='Perfect correlation')
plt.xlabel('LLM Predicted Z-Score')
plt.ylabel('Patient Actual Z-Score')
plt.title('Z-Score Correlation')
plt.legend()
plt.grid(True, alpha=0.3)

# Rank correlation plot
plt.subplot(2, 3, 2)
plt.scatter(merged['rank_llm'], merged['rank_patient'], color='green', alpha=0.7, s=50)
plt.xlabel('LLM Predicted Rank')
plt.ylabel('Patient Actual Rank')
plt.title('Rank Correlation')
plt.grid(True, alpha=0.3)

# Parametric comparison
plt.subplot(2, 3, 3)
plt.plot(x_llm.flatten(), y_llm, 'o-', label='LLM Predictions', color='blue')
plt.plot(x_llm.flatten(), y_patient, 's-', label='Patient Data', color='red')
plt.xlabel('Rank')
plt.ylabel('Z-Score Atrophy')
plt.title('Parametric Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Residual plot
plt.subplot(2, 3, 4)
residuals = y_patient - y_pred
plt.scatter(y_pred, residuals, color='purple', alpha=0.7, s=50)
plt.axhline(0, linestyle='--', color='r')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

# Regression line
plt.subplot(2, 3, 5)
plt.scatter(x_llm.flatten(), y_patient, color='blue', alpha=0.7, s=50)
plt.plot(x_llm.flatten(), y_pred, 'r-', linewidth=2, label=f'Regression (R²={r2:.3f})')
plt.xlabel('LLM Predicted Rank')
plt.ylabel('Patient Z-Score')
plt.title('Regression Analysis')
plt.legend()
plt.grid(True, alpha=0.3)

# Residuals histogram
plt.subplot(2, 3, 6)
plt.hist(residuals, bins=10, color='orange', alpha=0.7, edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title(f'Error Distribution (RMSE={rmse:.3f})')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print statistical summary
print("="*80)
print("📈 ANALYSIS RESULTS")
print("="*80)

z_score_corr = np.corrcoef(y_llm, y_patient)[0, 1]
rank_corr = np.corrcoef(merged['rank_llm'], merged['rank_patient'])[0, 1]

print(f"Z-Score Correlation: {z_score_corr:.4f}")
print(f"Rank Correlation: {rank_corr:.4f}")
print(f"Regression Model: Patient_Z_Score = {reg.coef_[0]:.4f} * LLM_Rank + {reg.intercept_:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
#%%
# Run extra statistical analysis comparing LLM and patient z-scores
stats_results = run_stats(merged['z_score_atrophy_llm'], merged['z_score_atrophy_patient'])
print("Pearson correlation:", stats_results['pearson'])
print("Spearman correlation:", stats_results['spearman'])
print("Paired t-test:", stats_results['ttest'])

# Generate Bland-Altman plot
bland_altman_plot(merged['z_score_atrophy_llm'], merged['z_score_atrophy_patient'])


# %%
