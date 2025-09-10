"""
Enhanced Pipeline Runner for Mill Analysis Project
Demonstrates how to use the split architecture with optional heatmap functionality
Now includes intelligent file selection option and dual-scan mode for proper heatmap analysis
"""

from main_processor import MillAnalysisProcessor


def run_mill_analysis(include_heatmap: bool = False, 
                     heatmap_type: str = 'enhanced',
                     use_file_selector: bool = False):
    """
    Run the complete mill analysis pipeline with optional features.
    
    Args:
        include_heatmap: Whether to include heatmap analysis (default: False for backward compatibility)
        heatmap_type: Type of heatmap ('standard' or 'enhanced')
        use_file_selector: If True, use intelligent file selection. If False, use hardcoded paths (default)
    
    Returns:
        Processing results dictionary
    """
    print("Starting Mill Analysis Pipeline...")
    print("Using split architecture: data_loader + noise_removal + scaling + alignment + visualization")
    
    if include_heatmap:
        print("+ heatmap_generator (ENABLED)")
        print("+ dual_scan_mode (ENABLED for proper heatmap comparison)")
    
    if use_file_selector:
        print("+ intelligent_file_selection (ENABLED)")
    else:
        print("+ hardcoded_file_paths (DEFAULT)")
    
    # Create the main processor with appropriate modes
    # Enable dual-scan mode when heatmap is requested for proper comparison
    dual_scan_mode = include_heatmap  # Dual-scan needed for meaningful heatmap
    
    processor = MillAnalysisProcessor(
        use_file_selector=use_file_selector,
        dual_scan_mode=dual_scan_mode
    )
    
    # Choose pipeline based on heatmap requirement
    if include_heatmap:
        # Run complete pipeline with dual-scan heatmap analysis
        print(f"\nRunning ENHANCED pipeline with {heatmap_type} heatmap analysis...")
        print("Note: Dual-scan mode enabled - will compare two different inner scans")
        
        results = processor.run_dual_scan_heatmap_analysis(
            target_ratio=1.0,
            heatmap_type=heatmap_type
        )
        
        if results and results.get('processing_successful', False):
            print("\n SUCCESS: Enhanced mill analysis with heatmap completed!")
            print("\nStandard Results:")
            print("- Noise removed and mill structure cleaned")
            print("- Proper scaling applied (1:1 ratio)")
            print("- Reference and inner scans aligned")
            print("- Comprehensive visualizations created")
            
            print("\nHeatmap Results:")
            if results.get('heatmap_included', False):
                heatmap_analysis = results['heatmap_results'].get('analysis', {})
                if heatmap_type in heatmap_analysis:
                    stats = heatmap_analysis[heatmap_type]
                    print(f"- {heatmap_type.title()} heatmap generated")
                    
                    if results.get('dual_scan_mode', False):
                        print("- Two different scans compared for wear analysis")
                    else:
                        print("- Same scan comparison (fallback mode)")
                        
                    print(f"- Mean wear distance: {stats.get('mean_distance', 0):.2f}mm")
                    print(f"- Max wear distance: {stats.get('max_distance', 0):.2f}mm")
                    print(f"- Points analyzed: {stats.get('num_points', 0):,}")
                    
                    if heatmap_type == 'enhanced':
                        print(f"- 11-stage color mapping with side view analysis")
                    else:
                        print(f"- 4-stage color mapping")
            
            print("\nOutput Locations:")
            print("- Standard analysis: output/visualizations/")
            print("- Heatmap analysis: output/heatmaps/")
            
            return results
        else:
            print("\n ERROR: Enhanced mill analysis with heatmap failed")
            return {'processing_successful': False}
    
    else:
        # Run standard pipeline (original functionality)
        print("\nRunning STANDARD pipeline...")
        success = processor.run_complete_pipeline(target_ratio=1.0)
        
        if success:
            print("\n SUCCESS: Standard mill analysis completed!")
            print("\nResults:")
            print("- Noise removed and mill structure cleaned")
            print("- Proper scaling applied (1:1 ratio)")
            print("- Reference and inner scan aligned")
            print("- Comprehensive visualizations created")
            
            print("\nOutput Locations:")
            print("- Analysis results: output/visualizations/")
            print("- Aligned scans: output/aligned_scans/")
            print("- Intermediate files: output/intermediate/")
            
            return {'processing_successful': True}
        else:
            print("\n ERROR: Standard mill analysis failed")
            return {'processing_successful': False}


def run_with_heatmap_analysis(heatmap_type: str = 'enhanced', use_file_selector: bool = False):
    """
    Convenience function to run pipeline with heatmap analysis.
    Automatically enables dual-scan mode for proper heatmap comparison.
    
    Args:
        heatmap_type: Type of heatmap ('standard' or 'enhanced')
        use_file_selector: If True, use intelligent file selection
    
    Returns:
        Processing results dictionary
    """
    return run_mill_analysis(
        include_heatmap=True, 
        heatmap_type=heatmap_type,
        use_file_selector=use_file_selector
    )


def run_with_intelligent_file_selection():
    """
    Convenience function to run pipeline with intelligent file selection.
    
    Returns:
        Processing results dictionary
    """
    return run_mill_analysis(use_file_selector=True)


def run_standard_pipeline():
    """
    Convenience function to run standard pipeline (backward compatible).
    
    Returns:
        Processing results dictionary
    """
    return run_mill_analysis()


if __name__ == "__main__":
    print("="*80)
    print("MILL ANALYSIS PIPELINE RUNNER")
    print("="*80)
    print("Available modes:")
    print("1. Standard pipeline (backward compatible)")
    print("2. Pipeline with intelligent file selection") 
    print("3. Pipeline with heatmap analysis (dual-scan mode)")
    print("4. Full pipeline (file selection + heatmap + dual-scan)")
    print("="*80)
    
    import sys
    
    # Command line argument handling
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "file_selector" or mode == "intelligent":
            print("Running with intelligent file selection...")
            results = run_with_intelligent_file_selection()
            
        elif mode == "heatmap":
            heatmap_type = sys.argv[2] if len(sys.argv) > 2 else 'enhanced'
            print(f"Running with {heatmap_type} heatmap analysis...")
            print("Note: This will enable dual-scan mode for proper comparison")
            results = run_with_heatmap_analysis(heatmap_type)
            
        elif mode == "full":
            print("Running full pipeline (file selection + heatmap + dual-scan)...")
            results = run_mill_analysis(
                include_heatmap=True, 
                heatmap_type='enhanced',
                use_file_selector=True
            )
            
        else:
            print("Running standard pipeline (default)...")
            results = run_standard_pipeline()
    
    else:
        # Default behavior - standard pipeline for backward compatibility
        print("Running standard pipeline (default)...")
        results = run_standard_pipeline()
    
    # Print final status
    if results and results.get('processing_successful', False):
        print("\n🎉 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    else:
        print("\n❌ PIPELINE EXECUTION FAILED!")
    
    print("\nUsage examples:")
    print("python src/run_pipeline.py                    # Standard pipeline")
    print("python src/run_pipeline.py file_selector      # With file selection")
    print("python src/run_pipeline.py heatmap           # With heatmap (dual-scan)")
    print("python src/run_pipeline.py full              # File selection + heatmap + dual-scan")
    print("\nNote: Heatmap mode automatically enables dual-scan for proper comparison")
