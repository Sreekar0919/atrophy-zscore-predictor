#%%

import os
from openai import OpenAI
from dotenv import load_dotenv
import json
import pandas as pd
from descriptions.clinical_notes_generator import ClinicalNotesGenerator
import math
from descriptions.schema import json_schema
from descriptions.compare_combined import (
    ZScoreComparator,
    validate_dataframe,
    smart_region_matcher,
    manual_region_fixes,
    apply_region_mapping,
    run_stats,
    bland_altman_plot
)
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Verify imports are working
print("Imports successful!")
print("Available functions:", [validate_dataframe, smart_region_matcher, manual_region_fixes, apply_region_mapping, run_stats, bland_altman_plot])

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'), override=True)

#%%
# Set up the client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

#%%
# Define the clinical note

# Initialize the generator
generator = ClinicalNotesGenerator()

# Generate a single note with specific patient characteristics
clinical_note = generator.generate_note(
    stage=3
    )

print(clinical_note.note_text)  

#clinical_note = "patient with svPPA"
 
#%%
#LLM prediction from clinical note
# Build the prompt


diagnosis_prompt = f"""
Given the following clinical note, generate a diagnosis for the patient.

Clinical Note:
"{clinical_note}"
"""

diagnosis_response = client.chat.completions.create(
    model="o4-mini-2025-04-16",
    max_completion_tokens=1000,
    temperature= 1,
    messages=[
        {"role": "system", "content": "You are a medical assistant specialized in neuroimaging analysis. Analyze the clinical note to come up with a diagnosis for the patient. Just output the diagnosis and severity of the diagnosis(use a severity scale typical of the disorder you diagnose), no other text."},
        {"role": "user", "content": diagnosis_prompt}
    ]
)

diagnosis = diagnosis_response.choices[0].message.content
print(diagnosis)

#%%

prompt = f"""
Given the following clinical note, generate a prediction of all brain regions with atrophy and their estimated z-score of atrophy based on the Destrieux atlas. Dont just use and increment of 0.5. Actually give your best guess. Output the results in JSON format conforming to the provided JSON Schema. Use the Destrieux atlas to predict the brain regions.

The patient has semantic Variant Primary Progressive Aphasia.

Destrieux Atlas:

G_and_S_occipital_inf
G_and_S_transv_frontopol
G_cingulate-Isthmus
G_cingulate-Main_part 
G_cuneus
G_front_inf-Opercular
G_front_inf-Orbital
G_front_inf-Triangul
G_front_middle
G_front_sup
G_and_S_frontomargin
G_Ins_lg_and_S_cent_ins
G_insular_short
G_oc-temp_lat-fusifor
G_oc-temp_med-Lingual
G_oc-temp_med-Parahip
G_occipital_middle
G_occipital_sup
G_orbital
G_and_S_paracentral
G_pariet_inf-Angular
G_pariet_inf-Supramar
G_parietal_sup
G_postcentral
G_precentral
G_precuneus
G_rectus
G_subcallosal
G_and_S_subcentral
G_temp_sup-G_T_transv
G_temp_sup-Lateral
G_temp_sup-Plan_polar
G_temp_sup-Plan_tempo
G_temporal_inf
G_temporal_middle
Lat_Fis-ant-Horizont
Lat_Fis-ant-Vertical
Lat_Fis-post
Medial_wall
Pole_occipital
Pole_temporal
S_calcarine
S_central
S_cingul-Marginalis
S_circular_insula_ant
S_circular_insula_inf
S_circular_insula_sup
S_collat_transv_ant
S_collat_transv_post
S_front_inf
S_front_middle
S_front_sup
S_interm_prim-Jensen
S_intrapariet_and_P_trans
S_occipital_ant
S_oc_middle_and_Lunatus
S_oc_sup_and_transversal
S_oc-temp_lat
S_oc-temp_med_and_Lingual
S_orbital_lateral
S_orbital_med-olfact
S_orbital-H_Shaped
S_parieto_occipital
S_pericallosal
S_postcentral
S_precentral-inf-part
S_precentral-sup-part
S_suborbital
S_subparietal
S_supracingulate 
S_temporal_inf
S_temporal_sup
S_temporal_transverse

Clinical Note:
"{clinical_note}"

JSON Schema:
{json_schema}
"""
# Prompt the LLM with the clinical note and the JSON schema

response = client.chat.completions.create(
    model="gpt-4o-mini",
    max_completion_tokens=16000,
    temperature= 1,
    logprobs=True,
    top_logprobs= 0,
    messages=[
        {"role": "system", "content": "You are a medical assistant specialized in neuroimaging analysis. When deciding the z-score, they do not have to have the same interval or even be different values. I want it to be a proper reflection of the severity based on the clinical note. Analyze the clinical note and predict which brain regions are likely to show atrophy based on the symptoms and clinical presentation. Use the Destrieux atlas to predict the brain regions. When giving predictions use the 2009 naming convention.  Make sure to output all the regions in the Destrieux atlas. Just output the JSON object, no other text."},
        {"role": "user", "content": prompt}
    ]
)

print("Raw LLM output:")
raw_content = response.choices[0].message.content
print(raw_content)
#%%

# Logprobs calculation and parsing for the regions and z-scores
tokens = response.choices[0].logprobs.content


# Extract JSON substring (removes code block markers if present)
json_str = re.search(r'\{[\s\S]*\}', raw_content)
if json_str:
    json_str = json_str.group(0)
    try:
        predictions = json.loads(json_str)
        print("\nParsed predictions:")
        print(predictions)
        
        # Get word-level logprobs for all fields
        print("\nWord-level logprobs:")
        # First, find where each value starts in the raw content
        value_positions = []
        for prediction in predictions['atrophy_predictions']:   
            for field, value in prediction.items():
                # Convert all values to strings for consistent handling
                value_str = str(value)
                # Find the position of this value in the raw content
                # For strings, look for quoted values; for numbers, look for unquoted
                if isinstance(value, str):
                    pos = json_str.find(f'"{value_str}"')
                else:
                    pos = json_str.find(value_str)
                if pos != -1:
                    value_positions.append((field, value_str, pos, pos + len(value_str)))
        
        # Sort by position to process tokens in order
        value_positions.sort(key=lambda x: x[2])
        
        # Process tokens to build value logprobs
        current_value = None
        current_field = None
        current_value_logprob = 0
        current_pos = 0
        reconstructed = ""
        value_logprobs = {}  # Will store as {field: {value: logprob}}
        value_token_parts = {}  # Will store as {field: {value: [(token, logprob), ...]}}
        
        # Track if we're in a number (for z-scores)
        in_number = False
        number_start_pos = 0
        current_token_parts = []
        
        # Initialize logprobs for all fields
        for field in ['rank', 'region', 'hemisphere', 'z_score_atrophy']:
            value_logprobs[field] = {}
            value_token_parts[field] = {}
        
        for tok in tokens:
            token = tok.token
            reconstructed += token
            
            # Special handling for z-scores (numbers)
            if current_field == 'z_score_atrophy':
                # Check if this token is part of a number
                if token.strip() in ['-', '.'] or token.strip().isdigit():
                    if not in_number:
                        in_number = True
                        number_start_pos = current_pos
                        current_value_logprob = tok.logprob
                        current_token_parts = [(token, tok.logprob)]
                    else:
                        current_value_logprob += tok.logprob
                        current_token_parts.append((token, tok.logprob))
                elif in_number:
                    # We've reached the end of the number
                    in_number = False
                    if current_value is not None:
                        value_logprobs[current_field][current_value] = current_value_logprob
                        value_token_parts[current_field][current_value] = current_token_parts
                        print(f"Completed {current_field}: {current_value} with total logprob: {current_value_logprob}")
                        print(f"  Parts: {current_token_parts}")
                    current_value = None
                    current_value_logprob = 0
                    current_token_parts = []
            
            # Check if this token starts a new value we're tracking
            for field, value, start, end in value_positions:
                if start <= current_pos < end:
                    if current_value != value:  # New value started
                        if current_value is not None and not in_number:
                            value_logprobs[current_field][current_value] = current_value_logprob
                            value_token_parts[current_field][current_value] = current_token_parts
                            print(f"Completed {current_field}: {current_value} with total logprob: {current_value_logprob}")
                            print(f"  Parts: {current_token_parts}")
                        current_value = value
                        current_field = field
                        if not in_number:
                            current_value_logprob = tok.logprob
                            current_token_parts = [(token, tok.logprob)]
                            print(f"Started new {field}: {value} with initial logprob: {tok.logprob}")
                    elif not in_number:  # Continuing current value (but not for numbers)
                        current_value_logprob += tok.logprob
                        current_token_parts.append((token, tok.logprob))
                        print(f"Added to {field}: {value}, new total: {current_value_logprob}")
                    break
            
            current_pos += len(token)
        
        # Add the last value if we're not in a number
        if current_value is not None and not in_number:
            if current_field not in value_logprobs:
                value_logprobs[current_field] = {}
            if current_field not in value_token_parts:
                value_token_parts[current_field] = {}
            value_logprobs[current_field][current_value] = current_value_logprob
            value_token_parts[current_field][current_value] = current_token_parts
            print(f"Completed final {current_field}: {current_value} with total logprob: {current_value_logprob}")
            print(f"  Parts: {current_token_parts}")
        # Create a DataFrame with the results
        results = []
        for pred in predictions['atrophy_predictions']:
            # Convert all values to strings for consistent lookup
            region_str = str(pred['region'])
            hemisphere_str = str(pred['hemisphere'])
            z_score_str = str(pred['z_score_atrophy'])
            
            row = {
                'rank': pred['rank'],
                'region': pred['region'],
                'region_confidence': math.exp(value_logprobs['region'].get(region_str, 0)),
                'hemisphere': pred['hemisphere'],
                'hemisphere_confidence': math.exp(value_logprobs['hemisphere'].get(hemisphere_str, 0)),
                'z_score': pred['z_score_atrophy'],
                'z_score_confidence': math.exp(value_logprobs['z_score_atrophy'].get(z_score_str, 0))
            }
            results.append(row)
        
        df = pd.DataFrame(results)
        print("\nResults Table:")
        print(df.to_string(index=False, float_format=lambda x: '{:.4f}'.format(x)))

    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print("Extracted string:", json_str)
else:
    print("No JSON object found in the output.")

#%%

# Generate a csv file with the predictions

raw_content = response.choices[0].message.content
# Extract JSON substring and parse it
json_str = re.search(r'\{[\s\S]*\}', raw_content)
if json_str:
    json_str = json_str.group(0)
    try:
        predictions = json.loads(json_str)
        
        # Create a DataFrame from the predictions
        results = []
        for pred in predictions['atrophy_predictions']:
            row = {
                'rank': pred['rank'],
                'region': pred['region'],
                'hemisphere': pred['hemisphere'],
                'z_score_atrophy': pred['z_score_atrophy']
            }
            results.append(row)
        
        df = pd.DataFrame(results)
        
        # Save to CSV
        output_dir = Path('synthetic_data')
        output_dir.mkdir(exist_ok=True)
        predictions_file = output_dir / 'llm_predictions.csv'
        df.to_csv(predictions_file, index=False)
        print(f"\nSaved predictions to {predictions_file}")
        
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print("Extracted string:", json_str)
else:
    print("No JSON object found in the output.")


# %%
# Compare predictions to ground truth

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
        
        # 7. Merge datasets
        merged = pd.merge(
            llm_df_mapped,
            patient_df,
            on=['region', 'hemisphere'],
            suffixes=('_llm', '_patient')
        )
                
        if len(merged) == 0:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_completion_tokens=16000,
                temperature=1,
                logprobs=True,
                top_logprobs=0,
                messages=[
                    {"role": "system", "content": "Compare these two lists to see if the unmatched regions from the LLM predictions are similar to the unmatched regions from the patient data. If they are, just return the matched regions as a JSON object with keys 'llm_region' and 'atlas_region'. Else, return an empty JSON object."},
                    {"role": "user", "content": f"""LLM predictions: {unmatched_llm}\nPatient data: {unmatched_atlas}
                    One example of a difficult region to match is:
                    G_oc-temp_med-Parahip and G_occipit-temp_med-Parahippocampal_part for instance.
                    """}
                ]
            )
            print(response.choices[0].message.content)
            # Parse the response and update the mapping
            try:
                matches = json.loads(response.choices[0].message.content)
                if matches:
                    # Update the final mapping with new matches
                    for match in matches:
                        final_mapping[match['llm_region']] = match['atlas_region']
                    
                    # Reapply mapping and merge with updated matches
                    llm_df_mapped = apply_region_mapping(llm_df, final_mapping, is_llm_data=True)
                    merged = pd.merge(
                        llm_df_mapped,
                        patient_df,
                        on=['region', 'hemisphere'],
                        suffixes=('_llm', '_patient')
                    )
            except json.JSONDecodeError:
                print("No additional matches found in LLM response")
        
        print(response.choices[0].message.content)
        print(f"\n🎯 FINAL MERGE RESULTS:")
        print(f"  Successfully matched: {len(merged)} region pairs")

    else:
        # All regions matched exactly
        merged = pd.merge(
            llm_df,
            patient_df,
            on=['region', 'hemisphere'],
            suffixes=('_llm', '_patient')
        )
    
    # 8. Extract variables for analysis
    x_llm = merged['rank_llm'].values.reshape(-1, 1)
    y_llm = merged['z_score_atrophy_llm'].values
    y_patient = merged['z_score_atrophy_patient'].values
    
    # Validate arrays
    if len(x_llm) < 2:
        raise ValueError("Insufficient data points for regression analysis")
    
    # 9. Run linear regression
    reg = LinearRegression().fit(x_llm, y_patient)
    y_pred = reg.predict(x_llm)
    r2 = reg.score(x_llm, y_patient)
    rmse = np.sqrt(mean_squared_error(y_patient, y_pred))
    
    # 10. Create comprehensive visualizations
    plt.figure(figsize=(18, 12))
    
    # Plot 1: Z-score correlation
    plt.subplot(2, 4, 1)
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
    plt.subplot(2, 4, 2)
    plt.scatter(merged['rank_llm'], merged['rank_patient'], alpha=0.7, color='green', s=50)
    plt.xlabel("LLM Predicted Rank")
    plt.ylabel("Patient Actual Rank")
    plt.title("Rank Correlation")
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Parametric curves
    plt.subplot(2, 4, 3)
    plt.plot(x_llm.flatten(), y_llm, 'o-', label='LLM Predictions', color='blue', alpha=0.7, markersize=6)
    plt.plot(merged['rank_patient'], y_patient, 's-', label='Patient Data', color='red', alpha=0.7, markersize=6)
    plt.xlabel("Rank")
    plt.ylabel("Z-Score Atrophy")
    plt.title("Parametric Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Residuals
    plt.subplot(2, 4, 4)
    residuals = y_patient - y_pred
    plt.scatter(y_pred, residuals, alpha=0.7, color='purple', s=50)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Regression line
    plt.subplot(2, 4, 5)
    plt.scatter(x_llm.flatten(), y_patient, alpha=0.7, color='blue', s=50)
    plt.plot(x_llm.flatten(), y_pred, 'r-', alpha=0.8, linewidth=2, label=f'Regression Line (R²={r2:.3f})')
    plt.xlabel("LLM Predicted Rank")
    plt.ylabel("Patient Actual Z-Score")
    plt.title("Regression Analysis")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Error distribution
    plt.subplot(2, 4, 6)
    plt.hist(residuals, bins=10, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title(f"Error Distribution (RMSE={rmse:.3f})")
    plt.grid(True, alpha=0.3)
    
    # Plot 7: Seaborn regression plot
    plt.subplot(2, 4, 7)
    sns.regplot(x='rank_llm', y='z_score_atrophy_patient', data=merged, 
                line_kws={'color': 'red'}, scatter_kws={'alpha': 0.7})
    plt.title("Seaborn Regression")
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Correlation heatmap
    plt.subplot(2, 4, 8)
    corr_data = merged[['rank_llm', 'z_score_atrophy_llm', 'rank_patient', 'z_score_atrophy_patient']]
    sns.heatmap(corr_data.corr(), annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
    plt.title("Correlation Matrix")
    
    plt.tight_layout()
    plt.show()
    
    # 11. Additional statistical analysis
    stats_results = run_stats(y_llm, y_patient)
    
    # 12. Bland-Altman plot
    bland_altman_plot(y_llm, y_patient)
    
    # 13. Print comprehensive results
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
    
    print(f"\n📊 ADVANCED STATISTICS:")
    print(f"  Pearson correlation: r={stats_results['pearson'][0]:.4f}, p={stats_results['pearson'][1]:.4f}")
    print(f"  Spearman correlation: rho={stats_results['spearman'][0]:.4f}, p={stats_results['spearman'][1]:.4f}")
    print(f"  Paired t-test: t={stats_results['ttest'][0]:.4f}, p={stats_results['ttest'][1]:.4f}")
    
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

    # 14. Compare z-scores
    zscore_comparator = ZScoreComparator('synthetic_data/llm_predictions.csv', 'synthetic_data/synthetic_data.csv')
    zscore_comparator.plot_comparison()
    report = zscore_comparator.generate_report()
    print(report)

except FileNotFoundError as e:
    print(f"❌ File Error: {str(e)}")
except ValueError as e:
    print(f"❌ Data Error: {str(e)}")
except Exception as e:
    print(f"❌ Unexpected Error: {str(e)}")
    import traceback
    traceback.print_exc()


# %%

# Compare decimal vs 2-digit integer z-score approaches
print("="*80)
print("COMPARING DECIMAL VS 2-DIGIT INTEGER Z-SCORE APPROACHES")
print("="*80)

# Approach 1: Original decimal z-scores
print("\n🔍 APPROACH 1: DECIMAL Z-SCORES (Original)")
print("-" * 50)

decimal_prompt = f"""
Given the following clinical note, generate a prediction of all brain regions with atrophy and their estimated z-score of atrophy based on the Destrieux atlas. Output the results in JSON format conforming to the provided JSON Schema. Use the Destrieux atlas to predict the brain regions.

The patient has semantic Variant Primary Progressive Aphasia.

Destrieux Atlas:
G_and_S_occipital_inf
G_and_S_transv_frontopol
G_cingulate-Isthmus
G_cingulate-Main_part 
G_cuneus
G_front_inf-Opercular
G_front_inf-Orbital
G_front_inf-Triangul
G_front_middle
G_front_sup
G_and_S_frontomargin
G_Ins_lg_and_S_cent_ins
G_insular_short
G_oc-temp_lat-fusifor
G_oc-temp_med-Lingual
G_oc-temp_med-Parahip
G_occipital_middle
G_occipital_sup
G_orbital
G_and_S_paracentral
G_pariet_inf-Angular
G_pariet_inf-Supramar
G_parietal_sup
G_postcentral
G_precentral
G_precuneus
G_rectus
G_subcallosal
G_and_S_subcentral
G_temp_sup-G_T_transv
G_temp_sup-Lateral
G_temp_sup-Plan_polar
G_temp_sup-Plan_tempo
G_temporal_inf
G_temporal_middle
Lat_Fis-ant-Horizont
Lat_Fis-ant-Vertical
Lat_Fis-post
Medial_wall
Pole_occipital
Pole_temporal
S_calcarine
S_central
S_cingul-Marginalis
S_circular_insula_ant
S_circular_insula_inf
S_circular_insula_sup
S_collat_transv_ant
S_collat_transv_post
S_front_inf
S_front_middle
S_front_sup
S_interm_prim-Jensen
S_intrapariet_and_P_trans
S_occipital_ant
S_oc_middle_and_Lunatus
S_oc_sup_and_transversal
S_oc-temp_lat
S_oc-temp_med_and_Lingual
S_orbital_lateral
S_orbital_med-olfact
S_orbital-H_Shaped
S_parieto_occipital
S_pericallosal
S_postcentral
S_precentral-inf-part
S_precentral-sup-part
S_suborbital
S_subparietal
S_supracingulate 
S_temporal_inf
S_temporal_sup
S_temporal_transverse

Clinical Note:
"{clinical_note}"

JSON Schema:
{json_schema}
"""

decimal_response = client.chat.completions.create(
    model="gpt-4o-mini",
    max_completion_tokens=16000,
    temperature=1,
    logprobs=True,
    top_logprobs=0,
    messages=[
        {"role": "system", "content": "You are a medical assistant specialized in neuroimaging analysis. When deciding the z-score, they do not have to have the same interval or even be different values. I want it to be a proper reflection of the severity based on the clinical note. Analyze the clinical note and predict which brain regions are likely to show atrophy based on the symptoms and clinical presentation. Use the Destrieux atlas to predict the brain regions. When giving predictions use the 2009 naming convention. Make sure to output all the regions in the Destrieux atlas. Just output the JSON object, no other text."},
        {"role": "user", "content": decimal_prompt}
    ]
)

# Approach 2: 2-digit integer z-scores
print("\n🔍 APPROACH 2: 2-DIGIT INTEGER Z-SCORES")
print("-" * 50)

integer_prompt = f"""
Given the following clinical note, generate a prediction of all brain regions with atrophy and their estimated z-score of atrophy based on the Destrieux atlas. Output the results in JSON format conforming to the provided JSON Schema. Use the Destrieux atlas to predict the brain regions.

IMPORTANT: Express z-scores as 2-digit integers by multiplying by 10. For example:
- -2.5 becomes -25
- -1.8 becomes -18
- -3.2 becomes -32

The patient has semantic Variant Primary Progressive Aphasia.

Destrieux Atlas:
G_and_S_occipital_inf
G_and_S_transv_frontopol
G_cingulate-Isthmus
G_cingulate-Main_part 
G_cuneus
G_front_inf-Opercular
G_front_inf-Orbital
G_front_inf-Triangul
G_front_middle
G_front_sup
G_and_S_frontomargin
G_Ins_lg_and_S_cent_ins
G_insular_short
G_oc-temp_lat-fusifor
G_oc-temp_med-Lingual
G_oc-temp_med-Parahip
G_occipital_middle
G_occipital_sup
G_orbital
G_and_S_paracentral
G_pariet_inf-Angular
G_pariet_inf-Supramar
G_parietal_sup
G_postcentral
G_precentral
G_precuneus
G_rectus
G_subcallosal
G_and_S_subcentral
G_temp_sup-G_T_transv
G_temp_sup-Lateral
G_temp_sup-Plan_polar
G_temp_sup-Plan_tempo
G_temporal_inf
G_temporal_middle
Lat_Fis-ant-Horizont
Lat_Fis-ant-Vertical
Lat_Fis-post
Medial_wall
Pole_occipital
Pole_temporal
S_calcarine
S_central
S_cingul-Marginalis
S_circular_insula_ant
S_circular_insula_inf
S_circular_insula_sup
S_collat_transv_ant
S_collat_transv_post
S_front_inf
S_front_middle
S_front_sup
S_interm_prim-Jensen
S_intrapariet_and_P_trans
S_occipital_ant
S_oc_middle_and_Lunatus
S_oc_sup_and_transversal
S_oc-temp_lat
S_oc-temp_med_and_Lingual
S_orbital_lateral
S_orbital_med-olfact
S_orbital-H_Shaped
S_parieto_occipital
S_pericallosal
S_postcentral
S_precentral-inf-part
S_precentral-sup-part
S_suborbital
S_subparietal
S_supracingulate 
S_temporal_inf
S_temporal_sup
S_temporal_transverse

Clinical Note:
"{clinical_note}"

JSON Schema:
{json_schema}
"""

integer_response = client.chat.completions.create(
    model="gpt-4o-mini",
    max_completion_tokens=16000,
    temperature=1,
    logprobs=True,
    top_logprobs=0,
    messages=[
        {"role": "system", "content": "You are a medical assistant specialized in neuroimaging analysis. When deciding the z-score, they do not have to have the same interval or even be different values. I want it to be a proper reflection of the severity based on the clinical note. Analyze the clinical note and predict which brain regions are likely to show atrophy based on the symptoms and clinical presentation. Use the Destrieux atlas to predict the brain regions. When giving predictions use the 2009 naming convention. Make sure to output all the regions in the Destrieux atlas. IMPORTANT: Express z-scores as 2-digit integers by multiplying by 10 (e.g., -2.5 becomes -25). Just output the JSON object, no other text."},
        {"role": "user", "content": integer_prompt}
    ]
)

# Parse and analyze both responses
def extract_zscore_logprobs(response, approach_name):
    """Extract z-score logprobs from response"""
    raw_content = response.choices[0].message.content
    tokens = response.choices[0].logprobs.content
    
    # Extract JSON
    json_str = re.search(r'\{[\s\S]*\}', raw_content)
    if not json_str:
        print(f"❌ No JSON found in {approach_name}")
        return None, None
    
    try:
        predictions = json.loads(json_str.group(0))
        zscore_logprobs = {}
        zscore_confidence = {}
        
        # Find z-score values in the JSON
        json_str = json_str.group(0)
        for pred in predictions['atrophy_predictions']:
            z_score = str(pred['z_score_atrophy'])
            pos = json_str.find(z_score)
            if pos != -1:
                # Find the token(s) corresponding to this z-score
                current_pos = 0
                total_logprob = 0
                token_count = 0
                
                for tok in tokens:
                    if pos <= current_pos < pos + len(z_score):
                        total_logprob += tok.logprob
                        token_count += 1
                    current_pos += len(tok.token)
                
                if token_count > 0:
                    zscore_logprobs[z_score] = total_logprob
                    zscore_confidence[z_score] = math.exp(total_logprob)
        
        return zscore_logprobs, zscore_confidence
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error in {approach_name}: {e}")
        return None, None

# Extract logprobs for both approaches
decimal_logprobs, decimal_confidence = extract_zscore_logprobs(decimal_response, "Decimal")
integer_logprobs, integer_confidence = extract_zscore_logprobs(integer_response, "Integer")

print("\n📊 COMPARISON RESULTS")
print("="*80)

# 1. Direct logprob comparison
print("\n1️⃣ DIRECT LOGPROB COMPARISON:")
print("-" * 40)

if decimal_logprobs and integer_logprobs:
    print("Decimal Z-Scores (Original):")
    for zscore, logprob in sorted(decimal_logprobs.items()):
        print(f"  {zscore}: {logprob:.6f}")
    
    print("\nInteger Z-Scores (2-digit):")
    for zscore, logprob in sorted(integer_logprobs.items()):
        print(f"  {zscore}: {logprob:.6f}")
    
    # Calculate statistics
    decimal_logprobs_list = list(decimal_logprobs.values())
    integer_logprobs_list = list(integer_logprobs.values())
    
    print(f"\n📈 STATISTICS:")
    print(f"  Decimal - Mean: {np.mean(decimal_logprobs_list):.6f}, Std: {np.std(decimal_logprobs_list):.6f}")
    print(f"  Integer - Mean: {np.mean(integer_logprobs_list):.6f}, Std: {np.std(integer_logprobs_list):.6f}")
    print(f"  Decimal - Min: {min(decimal_logprobs_list):.6f}, Max: {max(decimal_logprobs_list):.6f}")
    print(f"  Integer - Min: {min(integer_logprobs_list):.6f}, Max: {max(integer_logprobs_list):.6f}")

# 2. Confidence score comparison
print("\n2️⃣ CONFIDENCE SCORE COMPARISON (exp(logprob)):")
print("-" * 40)

if decimal_confidence and integer_confidence:
    print("Decimal Z-Scores (Original):")
    for zscore, conf in sorted(decimal_confidence.items()):
        print(f"  {zscore}: {conf:.6f}")
    
    print("\nInteger Z-Scores (2-digit):")
    for zscore, conf in sorted(integer_confidence.items()):
        print(f"  {zscore}: {conf:.6f}")
    
    # Calculate statistics
    decimal_conf_list = list(decimal_confidence.values())
    integer_conf_list = list(integer_confidence.values())
    
    print(f"\n📈 STATISTICS:")
    print(f"  Decimal - Mean: {np.mean(decimal_conf_list):.6f}, Std: {np.std(decimal_conf_list):.6f}")
    print(f"  Integer - Mean: {np.mean(integer_conf_list):.6f}, Std: {np.std(integer_conf_list):.6f}")
    print(f"  Decimal - Min: {min(decimal_conf_list):.6f}, Max: {max(decimal_conf_list):.6f}")
    print(f"  Integer - Min: {min(integer_conf_list):.6f}, Max: {max(integer_conf_list):.6f}")

# 3. Variance/consistency analysis
print("\n3️⃣ VARIANCE/CONSISTENCY ANALYSIS:")
print("-" * 40)

if decimal_logprobs and integer_logprobs:
    decimal_values = list(decimal_logprobs.values())
    integer_values = list(integer_logprobs.values())
    
    print(f"📊 VARIANCE ANALYSIS:")
    print(f"  Decimal Z-Scores:")
    print(f"    Variance: {np.var(decimal_values):.6f}")
    print(f"    Coefficient of Variation: {np.std(decimal_values)/np.mean(decimal_values):.6f}")
    print(f"    Range: {max(decimal_values) - min(decimal_values):.6f}")
    
    print(f"  Integer Z-Scores:")
    print(f"    Variance: {np.var(integer_values):.6f}")
    print(f"    Coefficient of Variation: {np.std(integer_values)/np.mean(integer_values):.6f}")
    print(f"    Range: {max(integer_values) - min(integer_values):.6f}")
    
    # Test for significant difference in variance
    from scipy import stats
    try:
        f_stat, p_value = stats.levene(decimal_values, integer_values)
        print(f"\n🔬 STATISTICAL TEST (Levene's test for equal variance):")
        print(f"  F-statistic: {f_stat:.6f}")
        print(f"  p-value: {p_value:.6f}")
        if p_value < 0.05:
            print("  ✅ Significant difference in variance between approaches")
        else:
            print("  ❌ No significant difference in variance between approaches")
    except:
        print("  ⚠️ Could not perform statistical test (insufficient data)")

# Visualization
print("\n📊 VISUALIZATION:")
print("-" * 40)

if decimal_logprobs and integer_logprobs:
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Logprob comparison
    plt.subplot(2, 3, 1)
    decimal_zscores = list(decimal_logprobs.keys())
    decimal_probs = list(decimal_logprobs.values())
    integer_zscores = list(integer_logprobs.keys())
    integer_probs = list(integer_logprobs.values())
    
    plt.scatter(range(len(decimal_probs)), decimal_probs, alpha=0.7, label='Decimal', color='blue')
    plt.scatter(range(len(integer_probs)), integer_probs, alpha=0.7, label='Integer', color='red')
    plt.xlabel('Z-Score Index')
    plt.ylabel('Log Probability')
    plt.title('Log Probability Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Confidence comparison
    plt.subplot(2, 3, 2)
    decimal_conf = list(decimal_confidence.values())
    integer_conf = list(integer_confidence.values())
    
    plt.scatter(range(len(decimal_conf)), decimal_conf, alpha=0.7, label='Decimal', color='blue')
    plt.scatter(range(len(integer_conf)), integer_conf, alpha=0.7, label='Integer', color='red')
    plt.xlabel('Z-Score Index')
    plt.ylabel('Confidence (exp(logprob))')
    plt.title('Confidence Score Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Distribution comparison
    plt.subplot(2, 3, 3)
    plt.hist(decimal_probs, bins=10, alpha=0.7, label='Decimal', color='blue', edgecolor='black')
    plt.hist(integer_probs, bins=10, alpha=0.7, label='Integer', color='red', edgecolor='black')
    plt.xlabel('Log Probability')
    plt.ylabel('Frequency')
    plt.title('Log Probability Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Box plot
    plt.subplot(2, 3, 4)
    data_to_plot = [decimal_probs, integer_probs]
    plt.boxplot(data_to_plot, labels=['Decimal', 'Integer'])
    plt.ylabel('Log Probability')
    plt.title('Log Probability Box Plot')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Scatter plot of z-score vs logprob
    plt.subplot(2, 3, 5)
    decimal_numeric = [float(z) for z in decimal_zscores]
    integer_numeric = [float(z) for z in integer_zscores]
    
    plt.scatter(decimal_numeric, decimal_probs, alpha=0.7, label='Decimal', color='blue')
    plt.scatter(integer_numeric, integer_probs, alpha=0.7, label='Integer', color='red')
    plt.xlabel('Z-Score Value')
    plt.ylabel('Log Probability')
    plt.title('Z-Score vs Log Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Confidence distribution
    plt.subplot(2, 3, 6)
    plt.hist(decimal_conf, bins=10, alpha=0.7, label='Decimal', color='blue', edgecolor='black')
    plt.hist(integer_conf, bins=10, alpha=0.7, label='Integer', color='red', edgecolor='black')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Confidence Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

print("\n✅ COMPARISON COMPLETE!")
print("="*80)

#%%
