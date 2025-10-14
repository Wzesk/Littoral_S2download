# Littoral Pipeline Test Framework

A comprehensive testing framework for the Littoral Processing Pipeline that enables performance benchmarking, quality assessment, and regression testing across multiple sites and pipeline steps.

## Overview

The test framework provides:
- **Unit testing** for individual pipeline steps
- **Performance benchmarking** across multiple sites
- **Quality metrics** collection and analysis
- **Regression testing** capabilities
- **Automated report generation** with visualizations
- **Flexible test configuration** via YAML files

## Directory Structure

```
test_framework/
├── config/                     # Test configuration files
│   ├── test_config_schema.yaml # Configuration schema documentation
│   ├── quick_smoke_test.yaml   # Quick smoke test configuration
│   ├── performance_benchmark.yaml # Performance benchmark test
│   └── partial_pipeline_test.yaml # Partial pipeline test
├── tests/                      # Test implementations
│   ├── __init__.py
│   └── step_testers.py        # Individual step testers
├── results/                    # Test results output
│   └── (generated test results)
├── test_runner.py             # Main test runner
├── visualize_results.py       # Results analysis and visualization
└── README.md                  # This file
```

## Quick Start

### 1. Run a Quick Smoke Test

```bash
cd /home/walter_littor_al/Littoral_S2download
python test_framework/test_runner.py --config test_framework/config/quick_smoke_test.yaml
```

### 2. List Available Test Configurations

```bash
python test_framework/test_runner.py --list-configs
```

### 3. Run Performance Benchmark

```bash
python test_framework/test_runner.py --config test_framework/config/performance_benchmark.yaml
```

### 4. Generate Performance Report

```bash
python test_framework/visualize_results.py --generate-report
```

## Test Configuration

Test configurations are defined in YAML files with the following structure:

```yaml
test_set:
  name: "test_name"
  description: "Test description"
  
  sites:
    - "Fenfushi"
    - "Bodufen"
    
  steps:
    - name: "download"
      enabled: true
      max_images: 5
      timeout: 300
    # ... more steps
    
  execution:
    parallel_sites: false
    cleanup_after: true
    save_intermediate: true
    
  output:
    detailed_logs: true
    performance_metrics: true
    quality_metrics: true
```

### Available Steps

- `download` - Image download from Earth Engine
- `coregister` - Image coregistration
- `cloud_impute` - Cloud imputation/removal
- `rgb_nir_creation` - RGB and NIR image creation
- `upsample` - Image upsampling
- `normalize` - Image normalization
- `segment` - Image segmentation
- `boundary_extract` - Shoreline boundary extraction
- `boundary_refine` - Boundary refinement
- `geotransform` - Coordinate transformation
- `filter_shorelines` - Shoreline filtering
- `tide_model` - Tidal modeling
- `tide_correct` - Tidal correction application

## Creating Custom Test Configurations

1. Copy an existing configuration file
2. Modify the `sites` list to include your target sites
3. Enable/disable steps as needed
4. Adjust timeouts and parameters
5. Save with a descriptive filename

Example minimal configuration:

```yaml
test_set:
  name: "my_custom_test"
  description: "Custom test for specific sites"
  
  sites:
    - "Fenfushi"
    
  steps:
    - name: "download"
      enabled: true
      max_images: 3
      timeout: 300
    - name: "segment"
      enabled: true
      timeout: 600
    - name: "boundary_extract"
      enabled: true
      timeout: 300
      
  execution:
    cleanup_after: true
    
  output:
    performance_metrics: true
```

## Test Results

Test results are saved in multiple formats:

### CSV Files
- `{test_name}_{timestamp}_summary.csv` - Site-level summary
- `{test_name}_{timestamp}_steps.csv` - Step-level details

### JSON Files
- `{test_name}_{timestamp}_detailed.json` - Complete test results

### Generated Reports
- Performance reports with visualizations
- Comparison reports between test runs

## Metrics Collected

### Performance Metrics
- Execution time per step
- Memory usage (initial, final, peak)
- CPU utilization
- File counts and sizes

### Quality Metrics (Step-specific)
- **Download**: Images downloaded, success rate
- **Coregistration**: Files processed, coregistration success rate
- **Cloud Imputation**: Cloud coverage statistics, processing success
- **Segmentation**: Mask creation success, file counts
- **Boundary Extraction**: Shoreline extraction success rates
- **Tide Modeling**: Tide range, prediction accuracy

### System Metrics
- Directory sizes
- File counts
- Error rates
- Processing throughput

## Analysis and Visualization

### Generate Performance Report

```bash
python test_framework/visualize_results.py --generate-report --results-dir test_framework/results/
```

### Compare Test Runs

```bash
python test_framework/visualize_results.py --compare baseline_results.csv new_results.csv
```

### Custom Analysis

The CSV results can be easily imported into analysis tools:

```python
import pandas as pd

# Load test results
summary_df = pd.read_csv('test_framework/results/test_summary.csv')
steps_df = pd.read_csv('test_framework/results/test_steps.csv')

# Analyze performance by site
site_performance = summary_df.groupby('site_name')['successful_steps'].mean()

# Analyze step durations
step_durations = steps_df.groupby('step_name')['duration_seconds'].agg(['mean', 'std'])
```

## Prerequisites

### Environment Setup
1. Ensure the `littoral_pipeline` conda environment is active
2. Cloud storage buckets must be mounted:
   ```bash
   python pipeline/mount_utils.py mount all
   ```

### Required Dependencies
- All pipeline dependencies (Earth Engine, processing libraries)
- pandas, numpy for data analysis
- matplotlib, seaborn for visualization (optional)

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the correct directory and the pipeline package is accessible
2. **Mount Errors**: Verify cloud storage buckets are mounted before testing
3. **Timeout Errors**: Increase timeout values in configuration for slow steps
4. **Memory Errors**: Reduce `max_images` parameter for memory-intensive tests

### Debug Mode

Run with detailed logging:

```bash
python test_framework/test_runner.py --config config.yaml --log-level DEBUG
```

### Checking Test Status

Monitor running tests:

```bash
# Check for running processes
ps aux | grep test_runner

# Monitor resource usage
htop
```

## Best Practices

### Test Design
- Start with small test sets (1-2 sites, few images)
- Use `partial_pipeline_test.yaml` to test specific steps
- Set reasonable timeouts based on expected processing times
- Enable `cleanup_after` for automated cleanup

### Performance Testing
- Use consistent hardware for benchmark comparisons
- Test with representative data sets
- Include multiple sites with varying characteristics
- Run tests multiple times for statistical significance

### Regression Testing
- Establish baseline performance metrics
- Test against known good configurations
- Compare results across code changes
- Document expected performance ranges

## Advanced Usage

### Custom Step Testers

To add testing for custom pipeline steps:

1. Create a new tester class in `tests/step_testers.py`
2. Inherit from `PipelineStepTester`
3. Implement `_execute_step()` method
4. Add to the tester registry in `test_runner.py`

### Integration with CI/CD

The test framework can be integrated into continuous integration:

```bash
# Run smoke test
python test_framework/test_runner.py --config test_framework/config/quick_smoke_test.yaml

# Check exit code
if [ $? -eq 0 ]; then
    echo "Tests passed"
else
    echo "Tests failed"
    exit 1
fi
```

### Automated Benchmarking

Set up automated benchmarking:

```bash
#!/bin/bash
# benchmark.sh

# Run benchmark
python test_framework/test_runner.py --config test_framework/config/performance_benchmark.yaml

# Generate report
python test_framework/visualize_results.py --generate-report

# Archive results
timestamp=$(date +%Y%m%d_%H%M%S)
mkdir -p benchmarks/$timestamp
cp -r test_framework/results/* benchmarks/$timestamp/
```

## Support

For issues or questions about the test framework:
1. Check the troubleshooting section above
2. Review test configuration syntax
3. Examine detailed error logs in test results
4. Verify pipeline dependencies and environment setup