import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ZScoreComparator:
    """Class for comparing and analyzing two sets of z-score predictions."""
    
    def __init__(self, file1: str, file2: str):
        """Initialize with paths to two CSV files containing z-score predictions."""
        self.df1 = pd.read_csv(file1)
        self.df2 = pd.read_csv(file2)
        self._validate_dataframes()
        
    def _validate_dataframes(self):
        """Validate that both dataframes have the required columns."""
        required_columns = ['region', 'hemisphere', 'z_score_atrophy', 'rank']
        for df, name in [(self.df1, 'First'), (self.df2, 'Second')]:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"{name} dataframe missing required columns: {missing_cols}")
    
    def get_region_overlap(self) -> Tuple[List[str], List[str], List[str]]:
        """Get lists of regions that overlap and are unique to each dataset."""
        regions1 = set(self.df1['region'].unique())
        regions2 = set(self.df2['region'].unique())
        
        overlap = list(regions1.intersection(regions2))
        unique_to_1 = list(regions1 - regions2)
        unique_to_2 = list(regions2 - regions1)
        
        return overlap, unique_to_1, unique_to_2
    
    def compare_z_scores(self) -> pd.DataFrame:
        """Compare z-scores for overlapping regions."""
        overlap, _, _ = self.get_region_overlap()
        
        # Create comparison dataframe
        comparison = []
        for region in overlap:
            for hemisphere in ['Left', 'Right']:
                z1 = self.df1[(self.df1['region'] == region) & 
                            (self.df1['hemisphere'] == hemisphere)]['z_score_atrophy'].values
                z2 = self.df2[(self.df2['region'] == region) & 
                            (self.df2['hemisphere'] == hemisphere)]['z_score_atrophy'].values
                
                if len(z1) > 0 and len(z2) > 0:
                    comparison.append({
                        'region': region,
                        'hemisphere': hemisphere,
                        'z_score_1': z1[0],
                        'z_score_2': z2[0],
                        'z_score_diff': z1[0] - z2[0],
                        'abs_diff': abs(z1[0] - z2[0])
                    })
        
        return pd.DataFrame(comparison)
    
    def analyze_rankings(self) -> pd.DataFrame:
        """Analyze differences in rankings between the two datasets."""
        overlap, _, _ = self.get_region_overlap()
        
        ranking_comparison = []
        for region in overlap:
            for hemisphere in ['Left', 'Right']:
                rank1 = self.df1[(self.df1['region'] == region) & 
                               (self.df1['hemisphere'] == hemisphere)]['rank'].values
                rank2 = self.df2[(self.df2['region'] == region) & 
                               (self.df2['hemisphere'] == hemisphere)]['rank'].values
                
                if len(rank1) > 0 and len(rank2) > 0:
                    ranking_comparison.append({
                        'region': region,
                        'hemisphere': hemisphere,
                        'rank_1': rank1[0],
                        'rank_2': rank2[0],
                        'rank_diff': rank1[0] - rank2[0],
                        'abs_rank_diff': abs(rank1[0] - rank2[0])
                    })
        
        return pd.DataFrame(ranking_comparison)
    
    def calculate_correlations(self) -> Dict[str, float]:
        """Calculate various correlation metrics between the two datasets."""
        comparison_df = self.compare_z_scores()
        
        correlations = {
            'pearson': stats.pearsonr(comparison_df['z_score_1'], comparison_df['z_score_2'])[0],
            'spearman': stats.spearmanr(comparison_df['z_score_1'], comparison_df['z_score_2'])[0],
            'kendall': stats.kendalltau(comparison_df['z_score_1'], comparison_df['z_score_2'])[0]
        }
        
        return correlations
    
    def plot_comparison(self, output_dir: str = 'comparison_plots'):
        """Generate basic plots to visualize the comparison."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 1. Scatter plot of z-scores
            plt.figure(figsize=(10, 8))
            comparison_df = self.compare_z_scores()
            if not comparison_df.empty:
                sns.scatterplot(data=comparison_df, x='z_score_1', y='z_score_2', hue='hemisphere')
                plt.plot([-5, 5], [-5, 5], 'k--', alpha=0.3)  # Diagonal line
                plt.title('Z-Score Comparison Between Datasets')
                plt.xlabel('Z-Score Dataset 1')
                plt.ylabel('Z-Score Dataset 2')
                plt.tight_layout()
                plt.savefig(output_path / 'zscore_scatter.png')
            plt.close()
            
            # 2. Heatmap of z-score differences
            if not comparison_df.empty:
                plt.figure(figsize=(12, 8))
                pivot_df = comparison_df.pivot(index='region', columns='hemisphere', values='z_score_diff')
                if not pivot_df.empty:
                    sns.heatmap(pivot_df, cmap='RdBu_r', center=0, annot=True, fmt='.2f')
                    plt.title('Z-Score Differences Between Datasets')
                    plt.tight_layout()
                    plt.savefig(output_path / 'zscore_differences_heatmap.png')
                plt.close()
            
            # 3. Lateralization comparison
            plt.figure(figsize=(8, 6))
            lateralization = self.calculate_lateralization()
            plt.bar(
                ['Dataset 1', 'Dataset 2'],
                [lateralization['patient1'], lateralization['patient2']]
            )
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            plt.title('Hemispheric Lateralization Comparison')
            plt.ylabel('Lateralization Score (Positive = Left Dominant)')
            plt.tight_layout()
            plt.savefig(output_path / 'lateralization_comparison.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generating plots: {str(e)}")
            raise
    
    def generate_report(self, output_file: str = 'comparison_report.txt'):
        """Generate a comprehensive report of the comparison analysis."""
        with open(output_file, 'w') as f:
            f.write("Z-Score Comparison Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic statistics
            f.write("Basic Statistics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Number of regions in dataset 1: {len(self.df1['region'].unique())}\n")
            f.write(f"Number of regions in dataset 2: {len(self.df2['region'].unique())}\n")
            
            # Region overlap analysis
            overlap, unique1, unique2 = self.get_region_overlap()
            f.write("\nRegion Overlap Analysis:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Number of overlapping regions: {len(overlap)}\n")
            if unique1:
                f.write(f"Regions unique to dataset 1: {', '.join(unique1)}\n")
            if unique2:
                f.write(f"Regions unique to dataset 2: {', '.join(unique2)}\n")
            
            # Correlation analysis
            correlations = self.calculate_correlations()
            f.write("\nCorrelation Analysis:\n")
            f.write("-" * 20 + "\n")
            for metric, value in correlations.items():
                f.write(f"{metric.capitalize()} correlation: {value:.3f}\n")
            
            # Z-score comparison statistics
            comparison_df = self.compare_z_scores()
            if not comparison_df.empty:
                f.write("\nZ-Score Comparison Statistics:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Mean absolute difference: {comparison_df['abs_diff'].mean():.3f}\n")
                f.write(f"Max absolute difference: {comparison_df['abs_diff'].max():.3f}\n")
                f.write(f"Standard deviation of differences: {comparison_df['z_score_diff'].std():.3f}\n")
                
                # Significant differences
                f.write("\nRegions with Significant Differences (>1.0 z-score):\n")
                f.write("-" * 20 + "\n")
                significant_diff = comparison_df[comparison_df['abs_diff'] > 1.0]
                if not significant_diff.empty:
                    for _, row in significant_diff.iterrows():
                        f.write(f"{row['region']} ({row['hemisphere']}): "
                               f"Difference = {row['z_score_diff']:.2f}\n")
                else:
                    f.write("No regions with significant differences found.\n")
            
            # Lateralization analysis
            lateralization = self.calculate_lateralization()
            f.write("\nLateralization Analysis:\n")
            f.write("-" * 20 + "\n")
            for patient, score in lateralization.items():
                dominance = "left" if score > 0 else "right"
                f.write(f"{patient}: {abs(score):.2f} {dominance}-dominant\n")
            
            # Overall similarity
            similarity = self.calculate_atrophy_similarity()
            f.write(f"\nOverall Pattern Similarity Score: {similarity:.3f}\n")
            f.write("(1.0 = identical patterns, 0.0 = completely different patterns)\n")

    def analyze_atrophy_patterns(self) -> pd.DataFrame:
        """Analyze the overall atrophy patterns between patients.
        
        This method categorizes regions into severity levels and compares the distribution
        of atrophy patterns between patients."""
        def categorize_severity(z_score: float) -> str:
            if z_score <= -2.0:
                return "Severe"
            elif z_score <= -1.0:
                return "Moderate"
            elif z_score <= -0.5:
                return "Mild"
            else:
                return "Normal"
        
        # Add severity categories to both dataframes
        df1 = self.df1.copy()
        df2 = self.df2.copy()
        df1['severity'] = df1['z_score_atrophy'].apply(categorize_severity)
        df2['severity'] = df2['z_score_atrophy'].apply(categorize_severity)
        
        # Create pattern comparison
        pattern_comparison = []
        for hemisphere in ['Left', 'Right']:
            for df, name in [(df1, 'patient1'), (df2, 'patient2')]:
                severity_counts = df[df['hemisphere'] == hemisphere]['severity'].value_counts()
                pattern_comparison.append({
                    'patient': name,
                    'hemisphere': hemisphere,
                    'severe_count': severity_counts.get('Severe', 0),
                    'moderate_count': severity_counts.get('Moderate', 0),
                    'mild_count': severity_counts.get('Mild', 0),
                    'normal_count': severity_counts.get('Normal', 0)
                })
        
        return pd.DataFrame(pattern_comparison)

    def calculate_lateralization(self) -> Dict[str, float]:
        """Calculate the degree of lateralization (left-right asymmetry) for each patient.
        
        Returns a dictionary with lateralization scores for each patient."""
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

    def calculate_atrophy_similarity(self) -> float:
        """Calculate a similarity score between the atrophy patterns of the two patients.
        
        This method uses a weighted combination of:
        1. Correlation between z-scores
        2. Similarity in regional dominance
        3. Similarity in lateralization patterns"""
        
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

    def plot_atrophy_patterns(self, output_dir: str = 'comparison_plots'):
        """Generate additional plots for atrophy pattern analysis."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Severity distribution plot
        plt.figure(figsize=(12, 6))
        pattern_df = self.analyze_atrophy_patterns()
        pattern_melted = pd.melt(
            pattern_df,
            id_vars=['patient', 'hemisphere'],
            value_vars=['severe_count', 'moderate_count', 'mild_count', 'normal_count'],
            var_name='severity',
            value_name='count'
        )
        sns.barplot(
            data=pattern_melted,
            x='patient',
            y='count',
            hue='severity',
            col='hemisphere'
        )
        plt.title('Distribution of Atrophy Severity by Hemisphere')
        plt.tight_layout()
        plt.savefig(output_path / 'severity_distribution.png')
        plt.close()
        
        # 2. Regional dominance comparison
        plt.figure(figsize=(10, 6))
        dominant_df = self.analyze_regional_dominance()
        sns.barplot(
            data=dominant_df,
            x='region',
            y='relative_severity',
            hue='patient'
        )
        plt.xticks(rotation=45, ha='right')
        plt.title('Relative Severity of Dominant Regions')
        plt.tight_layout()
        plt.savefig(output_path / 'regional_dominance.png')
        plt.close()
        
        # 3. Lateralization comparison
        plt.figure(figsize=(8, 6))
        lateralization = self.calculate_lateralization()
        plt.bar(
            ['Patient 1', 'Patient 2'],
            [lateralization['patient1'], lateralization['patient2']]
        )
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.title('Hemispheric Lateralization Comparison')
        plt.ylabel('Lateralization Score (Positive = Left Dominant)')
        plt.tight_layout()
        plt.savefig(output_path / 'lateralization_comparison.png')
        plt.close()

def main():
    """Example usage of the enhanced ZScoreComparator."""
    # Example usage
    comparator = ZScoreComparator('output/all_predictions.csv', 'output/all_predictions_2.csv')
    
    # Generate all visualizations
    comparator.plot_comparison()
    comparator.plot_atrophy_patterns()
    
    # Generate enhanced report
    comparator.generate_report()
    
    # Print key statistics
    logger.info("Analysis complete! Key findings:")
    
    # Pattern similarity
    similarity = comparator.calculate_atrophy_similarity()
    logger.info(f"Overall pattern similarity: {similarity:.3f}")
    
    # Lateralization
    lateralization = comparator.calculate_lateralization()
    logger.info("Lateralization scores:")
    for patient, score in lateralization.items():
        logger.info(f"{patient}: {score:.2f}")
    
    # Regional dominance
    dominant_regions = comparator.analyze_regional_dominance()
    logger.info("\nDominant regions by patient:")
    for patient in ['patient1', 'patient2']:
        regions = dominant_regions[dominant_regions['patient'] == patient]
        logger.info(f"{patient}: {', '.join(regions['region'].tolist())}")

if __name__ == "__main__":
    main() 