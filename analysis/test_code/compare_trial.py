import pandas as pd
from descriptions.compare import ZScoreComparator
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    
    # Initialize the comparator with the correct paths to the CSV files
    llm_predictions = script_dir / 'synthetic_data' / 'llm_predictions.csv'
    synthetic_data = script_dir / 'synthetic_data' / 'synthetic_data.csv'
    
    logger.info(f"Comparing files:\n{llm_predictions}\n{synthetic_data}")
    
    comparator = ZScoreComparator(str(llm_predictions), str(synthetic_data))
    
    # (Optional) Print overlapping regions and their ranks (for debugging)
    overlap, unique_llm, unique_synth = comparator.get_region_overlap()
    logger.info("Overlapping regions (and ranks) are:")
    for reg in overlap:
         llm_row = comparator.df1[comparator.df1['region'] == reg].iloc[0] if not comparator.df1[comparator.df1['region'] == reg].empty else None
         synth_row = comparator.df2[comparator.df2['region'] == reg].iloc[0] if not comparator.df2[comparator.df2['region'] == reg].empty else None
         if llm_row is not None and synth_row is not None:
             logger.info(f"Region: {reg} (LLM rank: {llm_row['rank']}, Synth rank: {synth_row['rank']})")
         else:
             logger.info(f"Region: {reg} (no rank info available)")
    
    # Generate comparison report (with a try block to handle fewer than 2 overlapping regions)
    logger.info("Generating comparison report...")
    report_path = script_dir / 'comparison_report.txt'
    try:
         comparator.generate_report(str(report_path))
    except ValueError as e:
         if "`x` and `y` must have length at least 2" in str(e):
             logger.warning("Not enough overlapping regions (at least 2) for correlation. Skipping report generation.")
         else:
             raise
    
    # Generate comparison plots (also wrapped in a try block)
    logger.info("Generating comparison plots...")
    plots_dir = script_dir / 'comparison_plots'
    try:
         comparator.plot_comparison(output_dir=str(plots_dir))
         comparator.plot_atrophy_patterns(output_dir=str(plots_dir))
    except Exception as e:
         logger.warning(f"Error generating plots (possibly due to no overlap): {e}")
    
    # Print key statistics (with a try block for correlation metrics)
    logger.info("\nComparison Analysis Summary:")
    logger.info("-" * 30)
    
    # Get region overlap information
    logger.info(f"\nRegion Overlap Analysis:")
    logger.info(f"Number of overlapping regions: {len(overlap)}")
    if unique_llm:
         logger.info(f"Regions unique to LLM predictions: {', '.join(unique_llm)}")
    if unique_synth:
         logger.info(f"Regions unique to synthetic data: {', '.join(unique_synth)}")
    
    # Get correlation metrics (with a try block)
    try:
         correlations = comparator.calculate_correlations()
         logger.info(f"\nCorrelation Analysis:")
         for metric, value in correlations.items():
             logger.info(f"{metric.capitalize()} correlation: {value:.3f}")
    except ValueError as e:
         if "`x` and `y` must have length at least 2" in str(e):
             logger.warning("Not enough overlapping regions (at least 2) for correlation metrics.")
         else:
             raise
    
    # Get atrophy pattern analysis (if available)
    try:
         patterns = comparator.analyze_atrophy_patterns()
         logger.info(f"\nAtrophy Pattern Analysis:")
         logger.info(patterns.to_string(index=False))
    except Exception as e:
         logger.warning(f"Error in atrophy pattern analysis (possibly due to no overlap): {e}")
    
    # Get lateralization analysis (if available)
    try:
         lateralization = comparator.calculate_lateralization()
         logger.info(f"\nLateralization Analysis:")
         for patient, score in lateralization.items():
             dominance = "left" if score > 0 else "right"
             logger.info(f"{patient}: {abs(score):.2f} {dominance}-dominant")
    except Exception as e:
         logger.warning(f"Error in lateralization analysis (possibly due to no overlap): {e}")
    
    # Calculate overall atrophy similarity (if available)
    try:
         similarity = comparator.calculate_atrophy_similarity()
         logger.info(f"\nOverall Atrophy Pattern Similarity: {similarity:.3f}")
         logger.info("(1.0 = identical patterns, 0.0 = completely different patterns)")
    except Exception as e:
         logger.warning(f"Error calculating atrophy similarity (possibly due to no overlap): {e}")

if __name__ == "__main__":
    main()
