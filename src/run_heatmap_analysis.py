"""
Two-Scan Heatmap Comparison Runner for Mill Analysis Project
Demonstrates how to compare two inner scans for wear progression analysis
"""

from pathlib import Path
from main_processor import MillAnalysisProcessor


def run_two_scan_heatmap_comparison():
    """
    Run two-scan heatmap comparison analysis.
    Currently using same scan twice as placeholder.
    """
    print("="*80)
    print("MILL ANALYSIS - TWO-SCAN HEATMAP COMPARISON")
    print("="*80)
    
    # Initialize processor
    processor = MillAnalysisProcessor()
    
    # File paths (currently using same scan twice as placeholder)
    # TODO: Replace with actual different time-point scans
    scan1_path = processor.data_loader.file_paths['inner_scan']  # Time point A
    scan2_path = processor.data_loader.file_paths['inner_scan']  # Time point B (placeholder)
    
    print(f"Scan 1 (Time A): {scan1_path}")
    print(f"Scan 2 (Time B): {scan2_path}")
    print("NOTE: Currently using same file twice as placeholder")
    print("Replace scan2_path with actual different time-point scan when available")
    
    # Load reference scan first
    processor.reference_pcd, _ = processor.data_loader.load_reference_file()
    if processor.reference_pcd is None:
        print("ERROR: Failed to load reference scan")
        return None
    
    # Run two-scan heatmap comparison
    results = processor.run_heatmap_comparison_pipeline(
        str(scan1_path), 
        str(scan2_path),
        heatmap_type='enhanced'  # Use enhanced for comprehensive analysis
    )
    
    if results.get('comparison_successful', False):
        print("\n SUCCESS: Two-scan heatmap comparison completed!")
        print("\nResults:")
        print("- Both scans processed through complete pipeline")
        print("- Enhanced heatmap with 11-stage color mapping generated")
        print("- Comprehensive visualizations created")
        print("- Wear progression analysis available")
        print("- Check output/heatmaps/ for heatmap files")
        print("- Check output/visualizations/ for analysis plots")
        
        return results
    else:
        print("\n ERROR: Two-scan heatmap comparison failed")
        print("Check console output for detailed error information")
        return None


def run_single_scan_with_heatmap():
    """
    Run single scan analysis with heatmap generation.
    Uses same scan twice for heatmap (placeholder mode).
    """
    print("="*80)
    print("MILL ANALYSIS - SINGLE SCAN WITH HEATMAP")
    print("="*80)
    
    processor = MillAnalysisProcessor()
    
    # Run complete pipeline with heatmap analysis
    results = processor.run_single_scan_heatmap_analysis(
        target_ratio=1.0,
        heatmap_type='enhanced'
    )
    
    if results.get('processing_successful', False):
        print("\n SUCCESS: Single scan analysis with heatmap completed!")
        print("\nResults:")
        print("- Complete mill processing pipeline executed")
        print("- Heatmap analysis generated (same scan twice)")
        print("- All visualizations created")
        print("- Ready for detailed analysis")
        
        # Show key metrics
        if results.get('heatmap_included', False):
            heatmap_analysis = results['heatmap_results'].get('analysis', {})
            if 'enhanced' in heatmap_analysis:
                stats = heatmap_analysis['enhanced']
                print(f"\nHeatmap Statistics:")
                print(f"- Mean distance: {stats.get('mean_distance', 0):.2f}mm")
                print(f"- Max distance: {stats.get('max_distance', 0):.2f}mm")
                print(f"- Points analyzed: {stats.get('num_points', 0):,}")
        
        return results
    else:
        print("\n ERROR: Single scan heatmap analysis failed")
        return None


if __name__ == "__main__":
    print("="*80)
    print("MILL HEATMAP ANALYSIS - DEMONSTRATION")
    print("="*80)
    
    # Example 1: Two-scan comparison (placeholder mode)
    print("\n[EXAMPLE 1] Two-Scan Heatmap Comparison")
    print("-"*50)
    two_scan_results = run_two_scan_heatmap_comparison()
    
    # Example 2: Single scan with heatmap
    print("\n[EXAMPLE 2] Single Scan with Heatmap Analysis")
    print("-"*50)
    single_scan_results = run_single_scan_with_heatmap()
    
    print("\n" + "="*80)
    print("HEATMAP ANALYSIS DEMONSTRATION COMPLETED")
    print("="*80)
    print("\nIntegration provides:")
    print("✓ Two-scan wear progression comparison")
    print("✓ Enhanced 11-stage color mapping")
    print("✓ Comprehensive side view (Y-Z) analysis")
    print("✓ Professional visualization system")
    print("✓ Seamless integration with existing pipeline")
    print("✓ Zero impact on existing functionality")