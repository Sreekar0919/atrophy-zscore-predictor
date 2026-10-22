# Smart Region Matcher with Advanced Statistical Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr, ttest_rel, kendalltau
from typing import Tuple, List, Dict, Optional
import os
from pathlib import Path
from difflib import SequenceMatcher
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
sns.set(style="whitegrid", palette="muted", font_scale=1.1)

def validate_dataframe(df, name, required_columns):
    """Validate dataframe has required columns and non-empty."""
    if df.empty:
        raise ValueError(f"{name} dataframe is empty")
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{name} dataframe missing required columns: {missing_cols}")
    return True

def clean_region_name(name: str) -> str:
    """Clean region name for better matching."""
    name = re.sub(r'\s+', ' ', name.strip())
    name = name.replace('_', '-')
    return name

def calculate_similarity(name1: str, name2: str) -> float:
    """Calculate similarity between two region names."""
    clean1 = clean_region_name(name1.lower())
    clean2 = clean_region_name(name2.lower())
    
    if clean1 == clean2:
        return 1.0
    
    # Sequence similarity
    seq_sim = SequenceMatcher(None, clean1, clean2).ratio()
    
    # Word-level similarity
    words1 = set(clean1.split('-'))
    words2 = set(clean2.split('-'))
    if words1 and words2:
        word_sim = len(words1.intersection(words2)) / len(words1.union(words2))
    else:
        word_sim = 0
    
    # Combined similarity (weighted)
    combined_sim = 0.6 * seq_sim + 0.4 * word_sim
    return combined_sim

def smart_region_matcher(llm_regions, atlas_regions, similarity_threshold=0.7):
    """Intelligently match region names using similarity scoring."""
    print("🧠 SMART REGION MATCHING")
    print("="*50)
    
    matches = {}
    unmatched_llm = []
    matched_atlas = set()
    
    # For each LLM region, find best atlas match
    for llm_region, llm_hem in llm_regions:
        best_match = None
        best_similarity = 0
        
        for atlas_region, atlas_hem in atlas_regions:
            # Only match within same hemisphere
            if llm_hem != atlas_hem:
                continue
                
            # Skip if already matched
            if (atlas_region, atlas_hem) in matched_atlas:
                continue
            
            similarity = calculate_similarity(llm_region, atlas_region)
            
            if similarity > best_similarity and similarity >= similarity_threshold:
                best_similarity = similarity
                best_match = (atlas_region, atlas_hem)
        
        if best_match:
            matches[(llm_region, llm_hem)] = best_match
            matched_atlas.add(best_match)
            print(f"✅ {llm_region} ({llm_hem}) → {best_match[0]} ({best_match[1]}) [sim: {best_similarity:.3f}]")
        else:
            unmatched_llm.append((llm_region, llm_hem))
            print(f"❌ {llm_region} ({llm_hem}) → No match found")
    
    # Find unmatched atlas regions
    unmatched_atlas = atlas_regions - matched_atlas
    
    print(f"\n📊 MATCHING SUMMARY:")
    print(f"  Successful matches: {len(matches)}")
    print(f"  Unmatched LLM regions: {len(unmatched_llm)}")
    print(f"  Unmatched Atlas regions: {len(unmatched_atlas)}")
    
    return matches, unmatched_llm, unmatched_atlas

def apply_region_mapping(df, mapping, is_llm_data=True):
    """Apply region mapping to dataframe."""
    df_mapped = df.copy()
    
    if is_llm_data:
        # For LLM data, map from LLM names to Atlas names
        region_map = {}
        for (llm_region, llm_hem), (atlas_region, atlas_hem) in mapping.items():
            region_map[(llm_region, llm_hem)] = (atlas_region, atlas_hem)
        
        # Apply mapping
        for i, row in df_mapped.iterrows():
            key = (row['region'], row['hemisphere'])
            if key in region_map:
                new_region, new_hem = region_map[key]
                df_mapped.at[i, 'region'] = new_region
                df_mapped.at[i, 'hemisphere'] = new_hem
    
    return df_mapped

def run_stats(y1, y2):
    """Run statistical tests comparing two arrays."""
    return {
        'pearson': pearsonr(y1, y2),
        'spearman': spearmanr(y1, y2),
        'ttest': ttest_rel(y1, y2)
    }

def bland_altman_plot(y1, y2):
    """Create Bland-Altman plot for agreement analysis."""
    mean = np.mean([y1, y2], axis=0)
    diff = y1 - y2
    md = np.mean(diff)
    sd = np.std(diff)

    plt.figure(figsize=(8, 6))
    plt.scatter(mean, diff, alpha=0.6, s=50)
    plt.axhline(md, color='gray', linestyle='--', label=f'Mean diff: {md:.3f}')
    plt.axhline(md + 1.96*sd, color='red', linestyle='--', label=f'+1.96 SD: {md + 1.96*sd:.3f}')
    plt.axhline(md - 1.96*sd, color='red', linestyle='--', label=f'-1.96 SD: {md - 1.96*sd:.3f}')
    plt.title('Bland-Altman Plot')
    plt.xlabel('Mean Z-Score')
    plt.ylabel('Difference (LLM - Patient)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def manual_region_fixes():
    """Manual fixes for common region name variations."""
    manual_mapping = {
        # Format: (llm_region, hemisphere): (atlas_region, hemisphere)
        # Add known mappings here based on specific mismatches
        ('G_temp_sup-Plan_polar', 'Left'): ('G_temp_sup-Plan_polar', 'Left'),
        ('G_temp_sup-Plan_polar', 'Right'): ('G_temp_sup-Plan_polar', 'Right'),
        ('G_temp_sup-Lateral', 'Left'): ('G_temp_sup-Lateral', 'Left'),
        ('G_temp_sup-Lateral', 'Right'): ('G_temp_sup-Lateral', 'Right'),
    }
    return manual_mapping

class ZScoreComparator:
    """Enhanced class for comparing and analyzing two sets of z-score predictions."""
    
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
            'pearson': pearsonr(comparison_df['z_score_1'], comparison_df['z_score_2'])[0],
            'spearman': spearmanr(comparison_df['z_score_1'], comparison_df['z_score_2'])[0],
            'kendall': kendalltau(comparison_df['z_score_1'], comparison_df['z_score_2'])[0]
        }
        
        return correlations
    
    def calculate_lateralization(self) -> Dict[str, float]:
        """Calculate the degree of lateralization (left-right asymmetry) for each patient."""
        def get_lateralization_score(df: pd.DataFrame) -> float:
            left_scores = df[df['hemisphere'] == 'Left']['z_score_atrophy'].values
            right_scores = df[df['hemisphere'] == 'Right']['z_score_atrophy'].values
            
            if len(left_scores) == 0 or len(right_scores) == 0:
                return 0.0
            
            left_mean = np.mean(left_scores)
            right_mean = np.mean(right_scores)
            return left_mean - right_mean
        
        return {
            'patient1': get_lateralization_score(self.df1),
            'patient2': get_lateralization_score(self.df2)
        }
    
    def analyze_regional_dominance(self, top_n: int = 3) -> pd.DataFrame:
        """Analyze which brain regions show the most dominant atrophy patterns."""
        def get_dominant_regions(df: pd.DataFrame) -> pd.DataFrame:
            region_means = df.groupby('region')['z_score_atrophy'].mean()
            dominant = region_means.nsmallest(top_n).reset_index()
            dominant['relative_severity'] = dominant['z_score_atrophy'] / region_means.min()
            return dominant
        
        patient1_dominant = get_dominant_regions(self.df1)
        patient2_dominant = get_dominant_regions(self.df2)
        
        patient1_dominant['patient'] = 'patient1'
        patient2_dominant['patient'] = 'patient2'
        return pd.concat([patient1_dominant, patient2_dominant], ignore_index=True)
    
    def calculate_atrophy_similarity(self) -> float:
        """Calculate a similarity score between the atrophy patterns."""
        z_correlation = self.calculate_correlations()['spearman']
        
        dominant_regions = self.analyze_regional_dominance()
        patient1_regions = set(dominant_regions[dominant_regions['patient'] == 'patient1']['region'])
        patient2_regions = set(dominant_regions[dominant_regions['patient'] == 'patient2']['region'])
        region_overlap = len(patient1_regions.intersection(patient2_regions)) / len(patient1_regions.union(patient2_regions))
        
        lateralization = self.calculate_lateralization()
        lat_diff = abs(lateralization['patient1'] - lateralization['patient2'])
        lat_similarity = 1.0 / (1.0 + lat_diff)
        
        weights = {'correlation': 0.4, 'regions': 0.4, 'lateralization': 0.2}
        similarity_score = (
            weights['correlation'] * z_correlation +
            weights['regions'] * region_overlap +
            weights['lateralization'] * lat_similarity
        )
        
        return similarity_score
    
    def plot_comparison(self, output_dir: str = 'comparison_plots'):
        """Generate comprehensive comparison plots."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 1. Z-score scatter plot
            plt.figure(figsize=(10, 8))
            comparison_df = self.compare_z_scores()
            if not comparison_df.empty:
                sns.scatterplot(data=comparison_df, x='z_score_1', y='z_score_2', hue='hemisphere')
                plt.plot([-5, 5], [-5, 5], 'k--', alpha=0.3)
                plt.title('Z-Score Comparison Between Datasets')
                plt.xlabel('Z-Score Dataset 1')
                plt.ylabel('Z-Score Dataset 2')
                plt.tight_layout()
                plt.savefig(output_path / 'zscore_scatter.png')
            plt.close()
            
            # 2. Z-score difference heatmap
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
            
            # 4. Regional dominance comparison
            plt.figure(figsize=(12, 6))
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
            
            # 5. Bland-Altman plot
            plt.figure(figsize=(8, 6))
            mean = np.mean([comparison_df['z_score_1'], comparison_df['z_score_2']], axis=0)
            diff = comparison_df['z_score_1'] - comparison_df['z_score_2']
            md = np.mean(diff)
            sd = np.std(diff)
            
            plt.scatter(mean, diff, alpha=0.6, s=50)
            plt.axhline(md, color='gray', linestyle='--', label=f'Mean diff: {md:.3f}')
            plt.axhline(md + 1.96*sd, color='red', linestyle='--', label=f'+1.96 SD: {md + 1.96*sd:.3f}')
            plt.axhline(md - 1.96*sd, color='red', linestyle='--', label=f'-1.96 SD: {md - 1.96*sd:.3f}')
            plt.title('Bland-Altman Plot')
            plt.xlabel('Mean Z-Score')
            plt.ylabel('Difference (Dataset 1 - Dataset 2)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path / 'bland_altman.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generating plots: {str(e)}")
            raise
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate a detailed comparison report."""
        report_lines = []
        
        # Header
        report_lines.append("Z-Score Comparison Analysis Report")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Basic statistics
        report_lines.append("Basic Statistics:")
        report_lines.append("-" * 20)
        report_lines.append(f"Number of regions in dataset 1: {len(self.df1['region'].unique())}")
        report_lines.append(f"Number of regions in dataset 2: {len(self.df2['region'].unique())}")
        report_lines.append("")
        
        # Region overlap analysis
        overlap, unique1, unique2 = self.get_region_overlap()
        report_lines.append("Region Overlap Analysis:")
        report_lines.append("-" * 20)
        report_lines.append(f"Number of overlapping regions: {len(overlap)}")
        if unique1:
            report_lines.append(f"Regions unique to dataset 1: {', '.join(unique1)}")
        if unique2:
            report_lines.append(f"Regions unique to dataset 2: {', '.join(unique2)}")
        report_lines.append("")
        
        # Correlation analysis
        correlations = self.calculate_correlations()
        report_lines.append("Correlation Analysis:")
        report_lines.append("-" * 20)
        for metric, value in correlations.items():
            report_lines.append(f"{metric.capitalize()} correlation: {value:.3f}")
        report_lines.append("")
        
        # Z-score comparison statistics
        comparison_df = self.compare_z_scores()
        if not comparison_df.empty:
            report_lines.append("Z-Score Comparison Statistics:")
            report_lines.append("-" * 20)
            report_lines.append(f"Mean absolute difference: {comparison_df['abs_diff'].mean():.3f}")
            report_lines.append(f"Max absolute difference: {comparison_df['abs_diff'].max():.3f}")
            report_lines.append(f"Standard deviation of differences: {comparison_df['z_score_diff'].std():.3f}")
            report_lines.append("")
            
            # Significant differences
            report_lines.append("Regions with Significant Differences (>1.0 z-score):")
            report_lines.append("-" * 20)
            significant_diff = comparison_df[comparison_df['abs_diff'] > 1.0]
            if not significant_diff.empty:
                for _, row in significant_diff.iterrows():
                    report_lines.append(f"{row['region']} ({row['hemisphere']}): "
                                      f"Difference = {row['z_score_diff']:.2f}")
            else:
                report_lines.append("No regions with significant differences found.")
            report_lines.append("")
        
        # Lateralization analysis
        lateralization = self.calculate_lateralization()
        report_lines.append("Lateralization Analysis:")
        report_lines.append("-" * 20)
        for patient, score in lateralization.items():
            dominance = "left" if score > 0 else "right"
            report_lines.append(f"{patient}: {abs(score):.2f} {dominance}-dominant")
        report_lines.append("")
        
        # Overall similarity
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