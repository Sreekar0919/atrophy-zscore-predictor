"""
Utility module for comparing atrophy predictions between patients.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from ..models.schema import AtrophyPredictions

logger = logging.getLogger(__name__)

class ZScoreComparator:
    """Utility class for comparing atrophy predictions between patients."""
    
    def __init__(self, predictions1: AtrophyPredictions, predictions2: AtrophyPredictions):
        """Initialize comparator with two sets of predictions.
        
        Args:
            predictions1: First set of atrophy predictions
            predictions2: Second set of atrophy predictions
        """
        self.df1 = predictions1.to_dataframe()
        self.df2 = predictions2.to_dataframe()
        
    def calculate_correlations(self) -> Dict[str, float]:
        """Calculate correlation between z-scores of the two predictions.
        
        Returns:
            Dictionary containing Pearson and Spearman correlation coefficients.
        """
        # Merge dataframes on region and hemisphere
        merged = pd.merge(
            self.df1, 
            self.df2,
            on=['region', 'hemisphere'],
            suffixes=('_1', '_2')
        )
        
        if merged.empty:
            logger.warning("No matching regions found for correlation")
            return {'pearson': 0.0, 'spearman': 0.0}
            
        return {
            'pearson': merged['z_score_atrophy_1'].corr(merged['z_score_atrophy_2'], method='pearson'),
            'spearman': merged['z_score_atrophy_1'].corr(merged['z_score_atrophy_2'], method='spearman')
        }
        
    def calculate_lateralization(self) -> Dict[str, float]:
        """Calculate the degree of lateralization (left-right asymmetry) for each patient.
        
        Returns:
            Dictionary with lateralization scores for each patient.
        """
        def get_lateralization_score(df: pd.DataFrame) -> float:
            left_scores = df[df['hemisphere'] == 'Left']['z_score_atrophy'].values
            right_scores = df[df['hemisphere'] == 'Right']['z_score_atrophy'].values
            
            if len(left_scores) == 0 or len(right_scores) == 0:
                return 0.0
            
            # Calculate mean atrophy for each hemisphere
            left_mean = np.mean(left_scores)
            right_mean = np.mean(right_scores)
            
            # Calculate lateralization score (positive = left dominant, negative = right dominant)
            return left_mean - right_mean
        
        return {
            'patient1': get_lateralization_score(self.df1),
            'patient2': get_lateralization_score(self.df2)
        }
        
    def analyze_regional_dominance(self, top_n: int = 3) -> pd.DataFrame:
        """Analyze which brain regions show the most dominant atrophy patterns.
        
        Args:
            top_n: Number of top regions to analyze
            
        Returns:
            DataFrame containing dominant regions for each patient.
        """
        def get_dominant_regions(df: pd.DataFrame) -> pd.DataFrame:
            # Calculate mean z-score for each region across hemispheres
            region_means = df.groupby('region')['z_score_atrophy'].mean()
            # Get top N most atrophied regions
            dominant = region_means.nsmallest(top_n).reset_index()
            dominant['relative_severity'] = dominant['z_score_atrophy'] / region_means.min()
            return dominant
        
        patient1_dominant = get_dominant_regions(self.df1)
        patient2_dominant = get_dominant_regions(self.df2)
        
        # Combine results
        patient1_dominant['patient'] = 'patient1'
        patient2_dominant['patient'] = 'patient2'
        return pd.concat([patient1_dominant, patient2_dominant], ignore_index=True)
        
    def calculate_atrophy_similarity(self) -> float:
        """Calculate a similarity score between the atrophy patterns of the two patients.
        
        Returns:
            Similarity score between 0 and 1, where 1 indicates identical patterns.
        """
        # 1. Z-score correlation (using Spearman to account for non-linear relationships)
        z_correlation = self.calculate_correlations()['spearman']
        
        # 2. Regional dominance similarity
        dominant_regions = self.analyze_regional_dominance()
        patient1_regions = set(dominant_regions[dominant_regions['patient'] == 'patient1']['region'])
        patient2_regions = set(dominant_regions[dominant_regions['patient'] == 'patient2']['region'])
        region_overlap = len(patient1_regions.intersection(patient2_regions)) / len(patient1_regions.union(patient2_regions))
        
        # 3. Lateralization similarity
        lateralization = self.calculate_lateralization()
        lat_diff = abs(lateralization['patient1'] - lateralization['patient2'])
        lat_similarity = 1.0 / (1.0 + lat_diff)  # Convert difference to similarity score
        
        # Combine scores with weights
        weights = {'correlation': 0.4, 'regions': 0.4, 'lateralization': 0.2}
        similarity_score = (
            weights['correlation'] * z_correlation +
            weights['regions'] * region_overlap +
            weights['lateralization'] * lat_similarity
        )
        
        return similarity_score
        
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate a detailed comparison report.
        
        Args:
            output_file: Optional path to save the report. If None, returns the report as a string.
            
        Returns:
            The report as a string if output_file is None.
        """
        report_lines = []
        
        # Add header
        report_lines.append("Atrophy Pattern Comparison Report")
        report_lines.append("=" * 40)
        report_lines.append("")
        
        # Add correlations
        correlations = self.calculate_correlations()
        report_lines.append("Correlation Analysis:")
        report_lines.append("-" * 20)
        report_lines.append(f"Pearson correlation: {correlations['pearson']:.3f}")
        report_lines.append(f"Spearman correlation: {correlations['spearman']:.3f}")
        report_lines.append("")
        
        # Add lateralization analysis
        lateralization = self.calculate_lateralization()
        report_lines.append("Lateralization Analysis:")
        report_lines.append("-" * 20)
        for patient, score in lateralization.items():
            dominance = "left" if score > 0 else "right"
            report_lines.append(f"{patient}: {abs(score):.2f} {dominance}-dominant")
        report_lines.append("")
        
        # Add regional dominance
        dominant_regions = self.analyze_regional_dominance()
        report_lines.append("Regional Dominance Analysis:")
        report_lines.append("-" * 20)
        for patient in ['patient1', 'patient2']:
            regions = dominant_regions[dominant_regions['patient'] == patient]
            report_lines.append(f"{patient} dominant regions:")
            for _, row in regions.iterrows():
                report_lines.append(f"  - {row['region']}: {row['z_score_atrophy']:.2f} (relative severity: {row['relative_severity']:.2f})")
        report_lines.append("")
        
        # Add overall similarity
        similarity = self.calculate_atrophy_similarity()
        report_lines.append("Overall Pattern Similarity:")
        report_lines.append("-" * 20)
        report_lines.append(f"Similarity score: {similarity:.3f}")
        report_lines.append("(1.0 = identical patterns, 0.0 = completely different patterns)")
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
                
        return report 