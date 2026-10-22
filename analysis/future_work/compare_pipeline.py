import os
from pathlib import Path
import logging
from typing import List, Tuple, Dict
import pandas as pd
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from pipeline_advanced import AtrophyPredictionPipeline
from compare import ZScoreComparator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComparisonPipeline:
    """Pipeline for generating and comparing atrophy predictions between patients."""
    
    def __init__(self, output_dir: str = "comparison_output"):
        """Initialize the pipeline with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline = AtrophyPredictionPipeline()
        
        # Create timestamped subdirectory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different types of output
        self.predictions_dir = self.run_dir / "predictions"
        self.comparison_dir = self.run_dir / "comparison"
        self.reports_dir = self.run_dir / "reports"
        for dir_path in [self.predictions_dir, self.comparison_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def generate_predictions(self, 
                           patient_notes: List[Tuple[str, str]], 
                           num_runs: int = 3) -> List[str]:
        """Generate multiple sets of predictions for each patient.
        
        Args:
            patient_notes: List of tuples containing (patient_id, clinical_note)
            num_runs: Number of prediction runs to perform for each patient
        
        Returns:
            List of paths to the generated prediction files
        """
        prediction_files = []
        
        for patient_id, note in patient_notes:
            logger.info(f"Generating predictions for patient {patient_id}")
            
            # Save the clinical note
            note_file = self.predictions_dir / f"{patient_id}_note.txt"
            with open(note_file, 'w') as f:
                f.write(f"Patient ID: {patient_id}\n")
                f.write("Clinical Note:\n")
                f.write(note)
            
            # Generate multiple sets of predictions
            for run in range(num_runs):
                try:
                    predictions = self.pipeline.predict_atrophy(note)
                    
                    # Save predictions as JSON
                    pred_file = self.predictions_dir / f"{patient_id}_run{run+1}.json"
                    predictions.save_json(str(pred_file))
                    prediction_files.append(str(pred_file))
                    
                    logger.info(f"Generated prediction run {run+1} for patient {patient_id}")
                    
                except Exception as e:
                    logger.error(f"Error generating prediction run {run+1} for patient {patient_id}: {str(e)}")
                    continue
        
        return prediction_files
    
    def combine_predictions(self, prediction_files: List[str]) -> pd.DataFrame:
        """Combine multiple prediction runs into a single DataFrame."""
        all_predictions = []
        
        for file in prediction_files:
            try:
                # Load predictions from JSON
                with open(file, 'r') as f:
                    data = json.load(f)
                
                # Convert to DataFrame
                df = pd.DataFrame(data['atrophy_predictions'])
                
                # Add metadata
                patient_id = Path(file).stem.split('_')[0]
                run_num = int(Path(file).stem.split('run')[1])
                df['patient_id'] = patient_id
                df['run'] = run_num
                
                all_predictions.append(df)
                
            except Exception as e:
                logger.error(f"Error processing prediction file {file}: {str(e)}")
                continue
        
        # Combine all predictions
        combined_df = pd.concat(all_predictions, ignore_index=True)
        
        # Save combined predictions
        combined_file = self.predictions_dir / "all_predictions.csv"
        combined_df.to_csv(combined_file, index=False)
        
        return combined_df
    
    def analyze_predictions(self, combined_df: pd.DataFrame):
        """Analyze the combined predictions to identify stable patterns."""
        # Group by patient and region to analyze consistency
        stability_analysis = combined_df.groupby(['patient_id', 'region', 'hemisphere']).agg({
            'z_score_atrophy': ['mean', 'std', 'count'],
            'rank': ['mean', 'std']
        }).round(3)
        
        # Save stability analysis
        stability_file = self.reports_dir / "prediction_stability.csv"
        stability_analysis.to_csv(stability_file)
        
        # Identify most stable predictions (low standard deviation)
        stable_predictions = stability_analysis[
            stability_analysis[('z_score_atrophy', 'std')] < 0.5
        ]
        
        # Save stable predictions
        stable_file = self.reports_dir / "stable_predictions.csv"
        stable_predictions.to_csv(stable_file)
        
        return stability_analysis, stable_predictions
    
    def compare_patients(self, 
                        patient1_id: str, 
                        patient2_id: str, 
                        combined_df: pd.DataFrame):
        """Compare atrophy patterns between two patients using stable predictions."""
        # Filter for the two patients
        patient1_df = combined_df[combined_df['patient_id'] == patient1_id]
        patient2_df = combined_df[combined_df['patient_id'] == patient2_id]
        
        # Calculate mean predictions for each patient
        patient1_mean = patient1_df.groupby(['region', 'hemisphere'])['z_score_atrophy'].mean().reset_index()
        patient2_mean = patient2_df.groupby(['region', 'hemisphere'])['z_score_atrophy'].mean().reset_index()
        
        # Save mean predictions
        patient1_file = self.comparison_dir / f"{patient1_id}_mean_predictions.csv"
        patient2_file = self.comparison_dir / f"{patient2_id}_mean_predictions.csv"
        patient1_mean.to_csv(patient1_file, index=False)
        patient2_mean.to_csv(patient2_file, index=False)
        
        # Perform comparison
        comparator = ZScoreComparator(str(patient1_file), str(patient2_file))
        
        # Generate all visualizations
        comparator.plot_comparison(output_dir=str(self.comparison_dir / "basic_comparison"))
        comparator.plot_atrophy_patterns(output_dir=str(self.comparison_dir / "pattern_analysis"))
        
        # Generate comprehensive report
        report_file = self.reports_dir / f"comparison_{patient1_id}_vs_{patient2_id}.txt"
        comparator.generate_report(output_file=str(report_file))
        
        return comparator
    
    def run_full_pipeline(self, 
                         patient_notes: List[Tuple[str, str]], 
                         num_runs: int = 3):
        """Run the complete pipeline from prediction generation to comparison."""
        logger.info("Starting full comparison pipeline")
        
        # 1. Generate predictions
        prediction_files = self.generate_predictions(patient_notes, num_runs)
        logger.info(f"Generated {len(prediction_files)} prediction files")
        
        # 2. Combine predictions
        combined_df = self.combine_predictions(prediction_files)
        logger.info(f"Combined predictions into DataFrame with {len(combined_df)} rows")
        
        # 3. Analyze prediction stability
        stability_analysis, stable_predictions = self.analyze_predictions(combined_df)
        logger.info("Completed stability analysis")
        
        # 4. Compare each pair of patients
        patient_ids = list(set(combined_df['patient_id']))
        for i in range(len(patient_ids)):
            for j in range(i + 1, len(patient_ids)):
                patient1_id = patient_ids[i]
                patient2_id = patient_ids[j]
                
                logger.info(f"Comparing patients {patient1_id} and {patient2_id}")
                comparator = self.compare_patients(patient1_id, patient2_id, combined_df)
                
                # Log key findings
                similarity = comparator.calculate_atrophy_similarity()
                logger.info(f"Pattern similarity between {patient1_id} and {patient2_id}: {similarity:.3f}")
        
        logger.info("Pipeline completed successfully")
        logger.info(f"Results saved in: {self.run_dir}")

    def load_real_patient_data(self, 
                             patient_id: str, 
                             data_file: str,
                             data_format: str = 'csv') -> pd.DataFrame:
        """Load real patient data from a file.
        
        Args:
            patient_id: Identifier for the patient
            data_file: Path to the file containing real patient data
            data_format: Format of the data file ('csv' or 'json')
            
        Returns:
            DataFrame containing the real patient data
        """
        try:
            if data_format.lower() == 'csv':
                df = pd.read_csv(data_file)
            elif data_format.lower() == 'json':
                with open(data_file, 'r') as f:
                    data = json.load(f)
                df = pd.DataFrame(data['atrophy_predictions'])
            else:
                raise ValueError(f"Unsupported data format: {data_format}")
            
            # Validate required columns
            required_columns = ['region', 'hemisphere', 'z_score_atrophy']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Real patient data missing required columns: {missing_cols}")
            
            # Add patient ID
            df['patient_id'] = patient_id
            df['data_source'] = 'real'
            
            # Save the real data
            output_file = self.predictions_dir / f"{patient_id}_real_data.csv"
            df.to_csv(output_file, index=False)
            
            logger.info(f"Loaded real patient data for {patient_id} from {data_file}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading real patient data for {patient_id}: {str(e)}")
            raise

    def compare_synthetic_vs_real(self,
                                patient_id: str,
                                clinical_note: str,
                                real_data_file: str,
                                num_synthetic_runs: int = 3,
                                real_data_format: str = 'csv') -> Dict:
        """Compare synthesized predictions with real patient data.
        
        Args:
            patient_id: Patient identifier
            clinical_note: Clinical note to generate synthetic predictions
            real_data_file: Path to file containing real patient data
            num_synthetic_runs: Number of synthetic prediction runs
            real_data_format: Format of the real data file
            
        Returns:
            Dictionary containing comparison metrics
        """
        logger.info(f"Starting synthetic vs real comparison for patient {patient_id}")
        
        # 1. Load real patient data
        real_df = self.load_real_patient_data(patient_id, real_data_file, real_data_format)
        
        # 2. Generate synthetic predictions
        synthetic_files = self.generate_predictions([(patient_id, clinical_note)], num_synthetic_runs)
        synthetic_df = self.combine_predictions(synthetic_files)
        
        # 3. Calculate mean synthetic predictions
        synthetic_mean = synthetic_df.groupby(['region', 'hemisphere'])['z_score_atrophy'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        
        # 4. Prepare comparison
        comparison_results = {
            'patient_id': patient_id,
            'metrics': {},
            'region_specific': []
        }
        
        # 5. Calculate overall metrics
        # Merge real and synthetic data
        merged_df = pd.merge(
            real_df,
            synthetic_mean,
            on=['region', 'hemisphere'],
            suffixes=('_real', '_synthetic')
        )
        
        # Calculate correlation metrics
        correlations = {
            'pearson': stats.pearsonr(merged_df['z_score_atrophy_real'], 
                                    merged_df['mean'])[0],
            'spearman': stats.spearmanr(merged_df['z_score_atrophy_real'], 
                                      merged_df['mean'])[0]
        }
        comparison_results['metrics']['correlations'] = correlations
        
        # Calculate error metrics
        merged_df['absolute_error'] = abs(merged_df['z_score_atrophy_real'] - merged_df['mean'])
        comparison_results['metrics'].update({
            'mean_absolute_error': merged_df['absolute_error'].mean(),
            'max_absolute_error': merged_df['absolute_error'].max(),
            'std_absolute_error': merged_df['absolute_error'].std()
        })
        
        # 6. Region-specific analysis
        for _, row in merged_df.iterrows():
            region_analysis = {
                'region': row['region'],
                'hemisphere': row['hemisphere'],
                'real_z_score': row['z_score_atrophy_real'],
                'synthetic_mean': row['mean'],
                'synthetic_std': row['std'],
                'absolute_error': row['absolute_error'],
                'synthetic_runs': row['count']
            }
            comparison_results['region_specific'].append(region_analysis)
        
        # 7. Generate visualizations
        self._plot_synthetic_vs_real_comparison(merged_df, patient_id)
        
        # 8. Generate detailed report
        self._generate_synthetic_vs_real_report(comparison_results, patient_id)
        
        return comparison_results

    def _plot_synthetic_vs_real_comparison(self, 
                                         merged_df: pd.DataFrame,
                                         patient_id: str):
        """Generate visualizations comparing synthetic and real data."""
        output_dir = self.comparison_dir / f"synthetic_vs_real_{patient_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Scatter plot with error bars
        plt.figure(figsize=(12, 8))
        plt.errorbar(
            merged_df['z_score_atrophy_real'],
            merged_df['mean'],
            yerr=merged_df['std'],
            fmt='o',
            capsize=5,
            label='Synthetic (mean ± std)'
        )
        plt.plot([-5, 5], [-5, 5], 'k--', alpha=0.3, label='Perfect agreement')
        plt.xlabel('Real Z-scores')
        plt.ylabel('Synthetic Z-scores')
        plt.title(f'Synthetic vs Real Z-scores for Patient {patient_id}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'zscore_comparison.png')
        plt.close()
        
        # 2. Error distribution by region
        plt.figure(figsize=(15, 8))
        sns.barplot(
            data=merged_df,
            x='region',
            y='absolute_error',
            hue='hemisphere'
        )
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Absolute Error by Region for Patient {patient_id}')
        plt.tight_layout()
        plt.savefig(output_dir / 'error_by_region.png')
        plt.close()
        
        # 3. Bland-Altman plot
        plt.figure(figsize=(10, 8))
        mean_values = (merged_df['z_score_atrophy_real'] + merged_df['mean']) / 2
        differences = merged_df['z_score_atrophy_real'] - merged_df['mean']
        
        plt.scatter(mean_values, differences, alpha=0.5)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axhline(y=differences.mean() + 1.96*differences.std(), color='r', linestyle='--', alpha=0.3)
        plt.axhline(y=differences.mean() - 1.96*differences.std(), color='r', linestyle='--', alpha=0.3)
        
        plt.xlabel('Mean of Real and Synthetic Z-scores')
        plt.ylabel('Difference (Real - Synthetic)')
        plt.title(f'Bland-Altman Plot for Patient {patient_id}')
        plt.tight_layout()
        plt.savefig(output_dir / 'bland_altman.png')
        plt.close()

    def _generate_synthetic_vs_real_report(self,
                                         comparison_results: Dict,
                                         patient_id: str):
        """Generate a detailed report comparing synthetic and real data."""
        report_file = self.reports_dir / f"synthetic_vs_real_{patient_id}.txt"
        
        with open(report_file, 'w') as f:
            f.write(f"Synthetic vs Real Data Comparison Report for Patient {patient_id}\n")
            f.write("=" * 60 + "\n\n")
            
            # Overall metrics
            f.write("Overall Comparison Metrics:\n")
            f.write("-" * 30 + "\n")
            f.write("Correlation Metrics:\n")
            for metric, value in comparison_results['metrics']['correlations'].items():
                f.write(f"  {metric.capitalize()}: {value:.3f}\n")
            
            f.write("\nError Metrics:\n")
            for metric, value in comparison_results['metrics'].items():
                if metric != 'correlations':
                    f.write(f"  {metric.replace('_', ' ').title()}: {value:.3f}\n")
            
            # Region-specific analysis
            f.write("\nRegion-Specific Analysis:\n")
            f.write("-" * 30 + "\n")
            for region in comparison_results['region_specific']:
                f.write(f"\nRegion: {region['region']} ({region['hemisphere']})\n")
                f.write(f"  Real Z-score: {region['real_z_score']:.2f}\n")
                f.write(f"  Synthetic Z-score: {region['synthetic_mean']:.2f} ± {region['synthetic_std']:.2f}\n")
                f.write(f"  Absolute Error: {region['absolute_error']:.2f}\n")
                f.write(f"  Number of synthetic runs: {region['synthetic_runs']}\n")
            
            # Interpretation
            f.write("\nInterpretation:\n")
            f.write("-" * 30 + "\n")
            pearson = comparison_results['metrics']['correlations']['pearson']
            mae = comparison_results['metrics']['mean_absolute_error']
            
            f.write("Correlation Strength:\n")
            if pearson > 0.7:
                f.write("  Strong correlation between synthetic and real data\n")
            elif pearson > 0.5:
                f.write("  Moderate correlation between synthetic and real data\n")
            else:
                f.write("  Weak correlation between synthetic and real data\n")
            
            f.write("\nError Analysis:\n")
            if mae < 0.5:
                f.write("  Low mean absolute error, suggesting good agreement\n")
            elif mae < 1.0:
                f.write("  Moderate mean absolute error, suggesting reasonable agreement\n")
            else:
                f.write("  High mean absolute error, suggesting significant differences\n")

def main():
    """Example usage of the comparison pipeline with real patient data."""
    # Example patient data
    patient_id = "P001"
    clinical_note = "Patient presents with progressive word-finding difficulties and behavioral changes over the past 14 months. Significant impact on daily functioning and quality of life."
    real_data_file = "path/to/real_patient_data.csv"  # Replace with actual path
    
    # Initialize pipeline
    pipeline = ComparisonPipeline()
    
    # Run synthetic vs real comparison
    try:
        results = pipeline.compare_synthetic_vs_real(
            patient_id=patient_id,
            clinical_note=clinical_note,
            real_data_file=real_data_file,
            num_synthetic_runs=3
        )
        
        # Print key findings
        logger.info(f"Comparison completed for patient {patient_id}")
        logger.info("Key metrics:")
        logger.info(f"Pearson correlation: {results['metrics']['correlations']['pearson']:.3f}")
        logger.info(f"Mean absolute error: {results['metrics']['mean_absolute_error']:.3f}")
        
    except Exception as e:
        logger.error(f"Error in synthetic vs real comparison: {str(e)}")
        raise

if __name__ == "__main__":
    main() 