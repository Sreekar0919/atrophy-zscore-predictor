"""
Schema definitions for brain atrophy predictions.
This module provides the JSON schema and Python classes for validating and working with atrophy predictions.
"""

from typing import Dict, Any, List, Optional, TypedDict, Literal
from dataclasses import dataclass
import json
from pathlib import Path

# Type definitions for better IDE support and type checking
Hemisphere = Literal["Left", "Right"]

class AtrophyPredictionDict(TypedDict):
    """Type definition for a single atrophy prediction"""
    rank: int
    region: str
    hemisphere: Hemisphere
    z_score_atrophy: float

class AtrophyPredictionsDict(TypedDict):
    """Type definition for a collection of atrophy predictions"""
    atrophy_predictions: List[AtrophyPredictionDict]

# JSON Schema definition
json_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "AtrophyPrediction",
    "type": "object",
    "properties": {
        "atrophy_predictions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rank": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Rank of the region in terms of atrophy severity (1 being most severe)"
                    },
                    "region": {
                        "type": "string",
                        "description": "Name of the brain region according to the Destrieux atlas"
                    },
                    "hemisphere": {
                        "type": "string",
                        "enum": ["Left", "Right"],
                        "description": "Hemisphere of the brain region"
                    },
                    "z_score_atrophy": {
                        "type": "number",
                        "description": "Z-score indicating the severity of atrophy (negative values indicate atrophy)"
                    }
                },
                "required": ["rank", "region", "hemisphere", "z_score_atrophy"],
                "additionalProperties": False
            },
            "minItems": 1,
            "description": "List of brain regions, ordered by severity of atrophy"
        }
    },
    "required": ["atrophy_predictions"],
    "additionalProperties": False
}

@dataclass
class AtrophyPrediction:
    """Class representing a single atrophy prediction"""
    rank: int
    region: str
    hemisphere: Hemisphere
    z_score_atrophy: float

    def __post_init__(self):
        """Validate the prediction after initialization"""
        if self.rank < 1:
            raise ValueError("Rank must be at least 1")
        if self.hemisphere not in ["Left", "Right"]:
            raise ValueError("Hemisphere must be either 'Left' or 'Right'")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AtrophyPrediction':
        """Create an AtrophyPrediction instance from a dictionary"""
        return cls(
            rank=data['rank'],
            region=data['region'],
            hemisphere=data['hemisphere'],
            z_score_atrophy=data['z_score_atrophy']
        )

    def to_dict(self) -> AtrophyPredictionDict:
        """Convert the prediction to a dictionary"""
        return {
            'rank': self.rank,
            'region': self.region,
            'hemisphere': self.hemisphere,
            'z_score_atrophy': self.z_score_atrophy
        }

class AtrophyPredictions:
    """Class representing a collection of atrophy predictions"""
    def __init__(self, predictions: Optional[List[AtrophyPrediction]] = None):
        self.predictions = predictions or []

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AtrophyPredictions':
        """Create an AtrophyPredictions instance from a dictionary"""
        predictions = [
            AtrophyPrediction.from_dict(pred)
            for pred in data['atrophy_predictions']
        ]
        return cls(predictions)

    def to_dict(self) -> AtrophyPredictionsDict:
        """Convert the predictions to a dictionary"""
        return {
            'atrophy_predictions': [
                pred.to_dict() for pred in self.predictions
            ]
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert the predictions to a JSON string"""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> 'AtrophyPredictions':
        """Create an AtrophyPredictions instance from a JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def save_json(self, filepath: str, indent: int = 2) -> None:
        """Save the predictions to a JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def load_json(cls, filepath: str) -> 'AtrophyPredictions':
        """Load predictions from a JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

def get_schema_json(indent: int = 2) -> str:
    """Get the JSON schema as a formatted string"""
    return json.dumps(json_schema, indent=indent)

def save_schema_json(filepath: Optional[str] = None) -> None:
    """Save the JSON schema to a file"""
    if filepath is None:
        filepath = Path(__file__).parent / 'schema.json'
    with open(filepath, 'w') as f:
        json.dump(json_schema, f, indent=2)

# Export the schema as JSON when this module is run directly
if __name__ == '__main__':
    save_schema_json() 