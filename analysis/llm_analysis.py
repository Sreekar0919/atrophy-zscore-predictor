#%%
from openai import OpenAI
import os
import pandas as pd

client = OpenAI()  # reads OPENAI_API_KEY from the environment

from pathlib import Path

script_dir = Path(__file__).parent

llm_predictions = script_dir / 'synthetic_data' / 'llm_predictions.csv'
synthetic_data = script_dir / 'synthetic_data' / 'synthetic_data.csv'

# Read the CSV files
llm_df = pd.read_csv(llm_predictions)
synthetic_df = pd.read_csv(synthetic_data)

# Convert DataFrames to string representation
llm_content = llm_df.to_string()
synthetic_content = synthetic_df.to_string()

#%%

diagnosis_prompt = f"""
You are evaluating the accuracy of an LLM's predictions for brain region atrophy analysis. 

TASK: Compare the LLM predictions against the ground truth data and provide a comprehensive accuracy assessment.

LLM Predictions (Top 5 most atrophied regions):
{llm_content}

Ground Truth Data (Complete brain region atrophy rankings):
{synthetic_content}

REQUIRED ANALYSIS:

1. REGION IDENTIFICATION ACCURACY:
   - How many of the LLM's predicted regions appear in the ground truth data?
   - Calculate precision: (correctly predicted regions) / (total LLM predictions)  
   - Calculate recall: (correctly predicted regions) / (total ground truth regions)
   - List exact matches and mismatches

2. Z-SCORE PREDICTION ACCURACY:
   - For regions that appear in both datasets, compare the z-score values
   - Calculate Mean Absolute Error (MAE) between LLM and ground truth z-scores
   - Identify which z-score predictions are most/least accurate

3. RANKING QUALITY ASSESSMENT:
   - Are the LLM's top-ranked regions actually among the most severe in ground truth?
   - For overlapping regions, compare their relative rankings
   - Calculate rank correlation (Spearman) if applicable

4. CLINICAL SIGNIFICANCE:
   - Which important high-atrophy regions (top 10 in ground truth) did the LLM miss?
   - Are there any LLM predictions that don't exist in ground truth (false positives)?
   - Overall assessment: Is the LLM capturing the most clinically relevant atrophied regions?

5. SUMMARY METRICS:
   - Overall accuracy score
   - Key strengths and weaknesses of the LLM predictions
   - Recommendations for improvement

Format your response with clear sections for each analysis point above, including specific numbers and percentages where applicable.
"""


diagnosis_response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    max_tokens=1000,
    temperature= 1,
    logprobs=True,
    top_logprobs= 0,
    messages=[
        {"role": "system", "content": "You are a medical assistant specialized in neuroimaging analysis. Analyze the two datasets and compare the predictions of the LLM and the synthetic data. Just output the comparison, no other text."},
        {"role": "user", "content": diagnosis_prompt}
    ]
)

print(diagnosis_response.choices[0].message.content)
# %%
