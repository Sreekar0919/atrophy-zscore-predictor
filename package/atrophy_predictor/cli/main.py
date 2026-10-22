"""
Command-line interface for the atrophy predictor package.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
from ..core.predictor import AtrophyPredictor
from ..models.schema import AtrophyPredictions
from ..utils.comparator import ZScoreComparator

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def predict_atrophy(args):
    """Handle the predict command."""
    predictor = AtrophyPredictor()
    
    # Read clinical note from file or stdin
    if args.input_file:
        with open(args.input_file, 'r') as f:
            clinical_note = f.read()
    else:
        clinical_note = sys.stdin.read()
    
    try:
        predictions = predictor.predict(clinical_note)
        
        # Output predictions
        if args.output_format == 'json':
            print(predictions.to_json())
        elif args.output_format == 'csv':
            predictions.to_dataframe().to_csv(sys.stdout, index=False)
        else:  # table format
            print("\nAtrophy Predictions:")
            print("=" * 60)
            print(predictions.to_dataframe().to_string(index=False))
            
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        sys.exit(1)

def compare_predictions(args):
    """Handle the compare command."""
    try:
        # Load predictions from files
        with open(args.file1, 'r') as f:
            predictions1 = AtrophyPredictions.from_json(f.read())
        with open(args.file2, 'r') as f:
            predictions2 = AtrophyPredictions.from_json(f.read())
            
        # Create comparator and generate report
        comparator = ZScoreComparator(predictions1, predictions2)
        report = comparator.generate_report(args.output_file)
        
        if not args.output_file:
            print(report)
            
    except Exception as e:
        logger.error(f"Error comparing predictions: {str(e)}")
        sys.exit(1)

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Atrophy Predictor - Predict and analyze brain atrophy patterns from clinical notes"
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Generate atrophy predictions from a clinical note')
    predict_parser.add_argument(
        '-i', '--input-file',
        help='Path to input file containing clinical note (default: read from stdin)'
    )
    predict_parser.add_argument(
        '-f', '--output-format',
        choices=['json', 'csv', 'table'],
        default='table',
        help='Output format for predictions (default: table)'
    )
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare atrophy predictions between two patients')
    compare_parser.add_argument(
        'file1',
        help='Path to first predictions file (JSON format)'
    )
    compare_parser.add_argument(
        'file2',
        help='Path to second predictions file (JSON format)'
    )
    compare_parser.add_argument(
        '-o', '--output-file',
        help='Path to save comparison report (default: print to stdout)'
    )
    
    # Global options
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    if args.command == 'predict':
        predict_atrophy(args)
    elif args.command == 'compare':
        compare_predictions(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main() 