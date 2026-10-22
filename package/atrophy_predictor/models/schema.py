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

@dataclass
class AtrophyPrediction:
    """A single atrophy prediction for a brain region."""
    rank: int
    region: str
    hemisphere: Hemisphere
    z_score_atrophy: float

    def to_dict(self) -> AtrophyPredictionDict:
        """Convert to dictionary format."""
        return {
            "rank": self.rank,
            "region": self.region,
            "hemisphere": self.hemisphere,
            "z_score_atrophy": self.z_score_atrophy
        }

    @classmethod
    def from_dict(cls, data: AtrophyPredictionDict) -> 'AtrophyPrediction':
        """Create from dictionary format."""
        return cls(**data)

@dataclass
class AtrophyPredictions:
    """A collection of atrophy predictions."""
    predictions: List[AtrophyPrediction]

    def to_dict(self) -> Dict[str, List[AtrophyPredictionDict]]:
        """Convert to dictionary format."""
        return {
            "atrophy_predictions": [p.to_dict() for p in self.predictions]
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    def to_dataframe(self) -> 'pd.DataFrame':
        """Convert to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame([p.to_dict() for p in self.predictions])

    @classmethod
    def from_dict(cls, data: Dict[str, List[AtrophyPredictionDict]]) -> 'AtrophyPredictions':
        """Create from dictionary format."""
        predictions = [AtrophyPrediction.from_dict(p) for p in data["atrophy_predictions"]]
        return cls(predictions=predictions)

    @classmethod
    def from_json(cls, json_str: str) -> 'AtrophyPredictions':
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

def get_schema_json() -> str:
    """Get the JSON schema for atrophy predictions."""
    schema = {
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
                            "minimum": 1
                        },
                        "region": {
                            "type": "string"
                        },
                        "hemisphere": {
                            "type": "string",
                            "enum": ["Left", "Right"]
                        },
                        "z_score_atrophy": {
                            "type": "number"
                        }
                    },
                    "required": ["rank", "region", "hemisphere", "z_score_atrophy"]
                }
            }
        },
        "required": ["atrophy_predictions"]
    }
    return json.dumps(schema, indent=2) 