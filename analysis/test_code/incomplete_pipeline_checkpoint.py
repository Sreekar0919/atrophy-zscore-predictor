#%%
import os
from openai import OpenAI
from dotenv import load_dotenv
import json
import pandas as pd
from descriptions.clinical_notes_generator import ClinicalNotesGenerator
import math
from descriptions.schema import json_schema
from descriptions.compare import ZScoreComparator

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'), override=True)

#%%
# Set up the client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

#%%
# Define the clinical note
'''
# Initialize the generator
generator = ClinicalNotesGenerator()

# Generate a single note with specific patient characteristics
clinical_note = generator.generate_note(
    patient_id="P0001",
    age=75,
    gender="Male",
    stage=3
    )

print(clinical_note.note_text)  

'''
clinical_note = "patient with svPPA"
 

# Build the prompt

prompt = f"""
Given the following clinical note, generate a prediction of the top 5 brain regions with atrophy and their estimated z-score of atrophy based on the Destrieux atlas. Output the results in JSON format conforming to the provided JSON Schema.

Clinical Note:
"{clinical_note}"

JSON Schema:
{json_schema}
"""

# Prompt the LLM with the clinical note and the JSON schema

response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    max_tokens=1000,
    temperature= 1,
    logprobs=True,
    top_logprobs= 1,
    messages=[
        {"role": "system", "content": "You are a medical assistant specialized in neuroimaging analysis. Analyze the clinical note and predict which brain regions are likely to show atrophy based on the symptoms and clinical presentation. Use the Destrieux atlas to predict the brain regions."},
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)
print(response.choices[0].logprobs.content)

#%%

# Calculate the logprobs for each token in the response



#%%
# Extract JSON and logprobs
response_text = response.choices[0].message.content
logprobs = response.choices[0].logprobs

# Find the first occurrence of '{' and the last occurrence of '}'
start_idx = response_text.find('{')
end_idx = response_text.rfind('}') + 1

if start_idx != -1 and end_idx != 0:
    json_str = response_text[start_idx:end_idx]
    try:
        response_data = json.loads(json_str)
        
        # Create DataFrame from the response data
        df = pd.DataFrame(response_data['atrophy_predictions'])
        
        # Sort by rank to ensure proper ordering
        df = df.sort_values('rank')
        
        # Display the DataFrame with some formatting
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        print("\nAtrophy Predictions DataFrame:")
        print(df.to_string(index=False))
        
        # Process and display log probabilities
        process_logprobs(logprobs, df)
        
        print(f"\nDataFrame shape: {df.shape}")
        print(f"DataFrame columns: {list(df.columns)}")
        
    except json.JSONDecodeError as e:
        print(f"\nError parsing JSON: {e}")
        print("Extracted JSON string:", json_str)
        raise
else:
    print("Could not find JSON data in the response")
    print("Full response:", response_text)

# %%
# Create synthetic "real" patient data for comparison
# This simulates what real patient data might look like for the same clinical note
synthetic_data = {
    'atrophy_predictions': [
        {'region': 'Superior temporal gyrus', 'hemisphere': 'Left', 'z_score_atrophy': -2.8, 'rank': 1},
        {'region': 'Middle temporal gyrus', 'hemisphere': 'Left', 'z_score_atrophy': -2.5, 'rank': 2},
        {'region': 'Inferior temporal gyrus', 'hemisphere': 'Left', 'z_score_atrophy': -2.2, 'rank': 3},
        {'region': 'Temporal pole', 'hemisphere': 'Left', 'z_score_atrophy': -2.0, 'rank': 4},
        {'region': 'Angular gyrus', 'hemisphere': 'Left', 'z_score_atrophy': -1.8, 'rank': 5},
    ]
}

# Create DataFrames for both datasets
llm_df = df.copy()  # The LLM predictions from previous cell
synthetic_df = pd.DataFrame(synthetic_data['atrophy_predictions'])

# Save both datasets to CSV files
llm_df.to_csv('llm_predictions.csv', index=False)
synthetic_df.to_csv('synthetic_data.csv', index=False)

# Import and use the ZScoreComparator
from descriptions.compare import ZScoreComparator

# Initialize the comparator
comparator = ZScoreComparator('llm_predictions.csv', 'synthetic_data.csv')

# Generate comparison report
comparator.generate_report('svppa_comparison_report.txt')

# Generate comparison plots
comparator.plot_comparison(output_dir='svppa_comparison_plots')

# Print some key statistics
print("\nComparison Analysis Summary:")
print("-" * 30)

# Get region overlap information
overlap, unique_llm, unique_synth = comparator.get_region_overlap()
print(f"\nRegion Overlap Analysis:")
print(f"Number of overlapping regions: {len(overlap)}")
print(f"Regions unique to LLM predictions: {', '.join(unique_llm)}")
print(f"Regions unique to synthetic data: {', '.join(unique_synth)}")

# Get correlation metrics
correlations = comparator.calculate_correlations()
print(f"\nCorrelation Analysis:")
for metric, value in correlations.items():
    print(f"{metric.capitalize()} correlation: {value:.3f}")

# Get atrophy pattern analysis
patterns = comparator.analyze_atrophy_patterns()
print(f"\nAtrophy Pattern Analysis:")
print(patterns.to_string(index=False))

# Get lateralization analysis
lateralization = comparator.calculate_lateralization()
print(f"\nLateralization Analysis:")
for patient, score in lateralization.items():
    print(f"{patient}: {score:.2f} (positive = left dominant, negative = right dominant)")

# Calculate overall atrophy similarity
similarity = comparator.calculate_atrophy_similarity()
print(f"\nOverall Atrophy Pattern Similarity: {similarity:.3f}")
# %%