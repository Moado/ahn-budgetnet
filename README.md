# AHN-BudgetNet: Cost-Aware Multimodal Feature-Acquisition Architecture for Parkinson's Disease Monitoring

[![DOI](https://img.shields.io/badge/DOI-10.3390%2Felectronics-blue)](https://doi.org/10.3390/electronics)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MDPI Electronics](https://img.shields.io/badge/Published%20in-MDPI%20Electronics-red)](https://www.mdpi.com/journal/electronics)

This repository contains the official implementation of **AHN-BudgetNet**, a cost-aware multimodal feature-acquisition architecture for Parkinson's disease monitoring, as published in *MDPI Electronics*.

## Abstract

AHN-BudgetNet introduces a novel framework for optimizing the cost-effectiveness of clinical feature acquisition in Parkinson's disease diagnosis and monitoring. Our approach systematically evaluates different combinations of clinical assessment tiers to maximize diagnostic performance while minimizing healthcare costs.

## Key Features

- **Multi-tier Cost Analysis**: Hierarchical evaluation of clinical assessments from basic demographics to advanced biomarkers
- **Efficiency Optimization**: Novel efficiency metrics balancing diagnostic performance against acquisition costs
- **Patient-level Cross-validation**: Robust evaluation preventing data leakage through GroupKFold validation
- **Comprehensive Sensitivity Analysis**: Parameter stability testing across multiple efficiency formulations
- **Break-even Analysis**: Economic viability assessment for high-cost diagnostic tiers
- **Clinical Decision Support**: Budget-aware recommendations for different healthcare scenarios

## Architecture Overview

AHN-BudgetNet organizes clinical features into five hierarchical tiers:

| Tier | Description | Cost (USD) | Time (min) | Features |
|------|-------------|------------|------------|----------|
| T0 | Demographics | $0 | 5 | Age, Education |
| T1 | Self-assessments | $75 | 30 | Cognitive self-reports, Motor scales |
| T2 | Clinical evaluations | $300 | 90 | Neurological assessments, Cognitive tests |
| T3 | DaTscan imaging | $3,300 | 180 | Dopamine transporter SPECT |
| T4 | Advanced biomarkers | $5,000 | 240 | Research-grade biomarkers |

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Dependencies

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn pathlib
```

### Quick Install

```bash
git clone https://github.com/Moado/ahn-budgetnet.git
cd ahn-budgetnet
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from ahn_budgetnet import DataLoader, AHNBudgetNet

# Load data
loader = DataLoader("your_dataset.xlsx")
data = loader.load_ppmi_data()

# Initialize framework
framework = AHNBudgetNet(data)

# Run comprehensive evaluation
results = framework.comprehensive_tier_evaluation()

# Generate report
report = framework.generate_comprehensive_report()
print(report)
```

### Advanced Configuration

```python
# Custom efficiency parameters
framework.efficiency_calculator.scaling_factors = [500, 1000, 2000]
framework.efficiency_calculator.epsilon_values = [0.05, 0.1, 0.2]

# Custom cross-validation
framework.cv_system = CrossValidationSystem(n_folds=5, random_state=123)

# Run sensitivity analysis
sensitivity_results = framework.perform_sensitivity_analysis()
```

## Dataset Requirements

The framework expects datasets with the following structure:

### Required Columns
- `PATIENT_ID`: Unique patient identifier
- `MOTOR_SEVERITY_RISK`: Binary target variable (0/1)
- Clinical features as specified in the tier hierarchy

### Supported Formats
- Excel files (.xlsx)
- CSV files (.csv)
- Pandas DataFrames

### Example Data Structure
```
PATIENT_ID | AGE_AT_VISIT | EDUCYRS | COGDXCL | ... | MOTOR_SEVERITY_RISK
P0001      | 65.2         | 16      | 1       | ... | 0
P0002      | 71.8         | 12      | 2       | ... | 1
...
```

## Key Components

### 1. DataLoader
Handles data loading with automatic fallback to synthetic data generation for testing and demonstration purposes.

### 2. EconomicHierarchy
Manages the tier-based feature organization and cost calculations with adaptation to available dataset features.

### 3. EfficiencyMetrics
Implements multiple efficiency formulations with comprehensive sensitivity analysis capabilities.

### 4. CrossValidationSystem
Provides patient-level cross-validation with confidence interval estimation to prevent data leakage.

### 5. AHNBudgetNet
Main framework class orchestrating the complete cost-aware feature selection pipeline.

## Efficiency Metrics

The framework implements several efficiency formulations:

### Primary Efficiency
```
E_primary = (AUC - baseline) / ((Cost/1000) + ε)
```

### Alternative Formulations
- **Logarithmic**: `E_log = (AUC - baseline) / log(Cost + 1)`
- **Square Root**: `E_sqrt = (AUC - baseline) / sqrt(Cost + 1)`
- **Clinical Utility**: `E_clinical = (AUC - baseline) × 1000 / (Cost + 50)`

## Results Interpretation

### Performance Metrics
- **AUC**: Area Under the ROC Curve with 95% confidence intervals
- **Efficiency**: Cost-normalized performance improvement
- **Cost-effectiveness**: Budget-specific recommendations

### Output Files
- `ahn_budgetnet_results.csv`: Detailed tier combination results
- Comprehensive analysis report with clinical recommendations

## Validation and Testing

The framework includes comprehensive validation:

- **Patient-level cross-validation** preventing data leakage
- **Sensitivity analysis** across parameter combinations
- **Statistical significance testing** with confidence intervals
- **Break-even analysis** for high-cost interventions

## Clinical Applications

### Budget Scenarios
The framework provides recommendations for different budget constraints:

- **Budget ≤$75**: Basic self-assessments
- **Budget ≤$300**: Clinical evaluations included
- **Budget ≤$1000**: Comprehensive clinical battery
- **Budget >$3000**: Advanced imaging considerations

### Decision Support
- Cost-effectiveness rankings for tier combinations
- Break-even analysis for expensive diagnostics
- Confidence intervals for performance estimates
- Literature-based performance benchmarks

## Contributing

We welcome contributions to improve AHN-BudgetNet. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

### Development Setup

```bash
git clone https://github.com/Moado/ahn-budgetnet.git
cd ahn-budgetnet
pip install -e .
pip install -r requirements-dev.txt
```

## Citation

If you use AHN-BudgetNet in your research, please cite our paper:

```bibtex
@article{ahn-budgetnet2025,
  title={AHN-BudgetNet: Cost-Aware Multimodal Feature-Acquisition Architecture for Parkinson's Disease Monitoring},
  author={[Moad Hani, Said Mahmoudi and Mohammed Benjelloun]},
  journal={Electronics},
  year={2025},
  publisher={MDPI},
  doi={10.3390/electronics},
  url={https://www.mdpi.com/journal/electronics}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **PPMI Study**: Data used in this work were obtained from the Parkinson's Progression Markers Initiative (PPMI) database
- **MDPI Electronics**: For publishing our research
- **Open Source Community**: For the excellent scientific Python ecosystem

## Contact

For questions, issues, or collaborations:

- **Email**: [moad.hani@umons.ac.be]
- **Issues**: [GitHub Issues](https://github.com/Moado/ahn-budgetnet/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Moado/ahn-budgetnet/discussions)

## Changelog

### Version 1.0.0 (2025-08-28)
- Initial release
- Complete AHN-BudgetNet framework implementation
- Comprehensive documentation and examples
- MDPI Electronics publication support

---

**Disclaimer**: This software is for research purposes only and should not be used for clinical decision-making without appropriate validation and regulatory approval.

## Technical Specifications

### System Requirements
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 1GB free space
- **Python**: 3.8+ with scientific computing stack

### Performance Benchmarks
- **Small datasets** (<1000 samples): <5 minutes execution
- **Medium datasets** (1000-5000 samples): 5-15 minutes execution  
- **Large datasets** (>5000 samples): 15-60 minutes execution

### Scalability
The framework is designed to handle datasets with:
- Up to 50,000 patients
- Up to 500 clinical features
- Multiple outcome variables
- Missing data patterns up to 95%
