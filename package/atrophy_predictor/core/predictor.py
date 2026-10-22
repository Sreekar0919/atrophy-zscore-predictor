"""
Core predictor module for generating atrophy predictions from clinical notes.
"""

import os
import logging
from typing import Optional
from anthropic import Anthropic
from ..models.schema import AtrophyPredictions, get_schema_json

logger = logging.getLogger(__name__)

class AtrophyPredictor:
    """Main class for predicting brain atrophy patterns from clinical notes."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the predictor with Anthropic API key.
        
        Args:
            api_key: Optional Anthropic API key. If not provided, will look for ANTHROPIC_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key must be provided or set as ANTHROPIC_API_KEY environment variable")
        
        self.client = Anthropic(api_key=self.api_key)
        
    def predict(self, clinical_note: str) -> AtrophyPredictions:
        """Generate atrophy predictions for a clinical note.
        
        Args:
            clinical_note: The clinical note text to analyze.
            
        Returns:
            AtrophyPredictions object containing the predicted atrophy patterns.
            
        Raises:
            ValueError: If the response cannot be parsed or is invalid.
        """
        prompt = f"""Given the following clinical note, generate a prediction of the top 5 brain regions with atrophy and their estimated z-score of atrophy based on the Destrieux atlas. Output the results in JSON format conforming to the provided JSON Schema.

Clinical Note:
"{clinical_note}"

JSON Schema:
{get_schema_json()}"""

        try:
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0,
                system="You are a medical assistant specialized in neuroimaging analysis. Analyze the clinical note and predict which brain regions are likely to show atrophy based on the symptoms and clinical presentation.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_content = response.content[0].text.strip()
            
            # Extract JSON from response
            start_idx = response_content.find('{')
            end_idx = response_content.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON content found in response")
                
            json_str = response_content[start_idx:end_idx]
            predictions = AtrophyPredictions.from_json(json_str)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            logger.error(f"Response content: {repr(response_content)}")
            raise 