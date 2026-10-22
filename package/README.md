# Atrophy Predictor

A Python package for predicting brain atrophy patterns from clinical notes, with a focus on semantic variant Primary Progressive Aphasia (svPPA).

## Installation

```bash
pip install atrophy-predictor
```

## Quick Start

```python
from atrophy_predictor import AtrophyPredictor

# Initialize the predictor
predictor = AtrophyPredictor()

# Predict atrophy from a clinical note
clinical_note = """
Patient presents with progressive word-finding difficulties and behavioral changes.
Symptoms began gradually 14 months ago with difficulty finding specific words during conversations.
Patient reports increasing frustration with word-finding difficulties.
"""

# Get predictions
predictions = predictor.predict(clinical_note)

# Print the predictions
print(predictions.to_dataframe())
```

## Features

- Predict brain atrophy patterns from clinical notes
- Compare atrophy patterns between patients
- Generate standardized reports
- Analyze lateralization and regional dominance
- Export predictions to various formats (DataFrame, JSON, CSV)

## Documentation

For detailed documentation, please visit [docs/](docs/).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 