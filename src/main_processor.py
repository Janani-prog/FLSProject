"""
Complete Mill Analysis Processing Pipeline
Open3D implementation - Orchestrates all split modules
Runs the complete V3 noise removal + old_version alignment pipeline
Now includes optional heatmap generation functionality and intelligent file selection
Enhanced with truly sequential dual-scan support for proper alignment
UPDATED: Source filename integration in visualization outputs for better traceability
"""

import open3d as o3d
import numpy as np
import copy
import time
import gc
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

from config import ProcessingConfig
from data_loader import PointCloudLoader
from noise_removal import NoiseRemovalProcessor
from scaling import ScalingProcessor
from alignment import AlignmentProcessor
from visualization import VisualizationProcessor
from heatmap_generator import WearHeatmapGenerator


class MillAnalysisProcessor:
    """Complete mill analysis processor combining all modules."""
    
    def __init__(self, use_file_selector: bool = False, dual_scan_mode: bool = False):
        """
        Initialize the processor.
        
        Args:
            use_file_selector: If True, use intelligent file selection. If False, use hardcoded paths (default)
            dual_scan_mode: If True, load two inner scans for heatmap comparison. If False, single scan (default)
        """
        self.config = ProcessingConfig()
        self.use_file_selector = use_file_selector
        self.dual_scan_mode = dual_scan_mode
        
        # Initialize all processing modules
        self.data_loader = PointCloudLoader()
        self.noise_processor = NoiseRemovalProcessor()
        self.scaling_processor = ScalingProcessor()
        
        # Updated: Use AlignmentProcessor with fixed scaling enabled
        self.alignment_processor = AlignmentProcessor(
            enable_fixed_scaling=True, fixed_scale_factor=1000.0
        )
        
        self.visualization_processor = VisualizationProcessor()
        
        # Core data storage
        self.reference_pcd = None
        
        # Scan 1 data
        self.inner_scan1_path = None
        self.inner_scan1_aligned = None
        self.scale_factor1 = 1.0
        
        # Scan 2 data  
        self.inner_scan2_path = None
        self.inner_scan2_aligned = None
        self.scale_factor2 = 1.0
        
        # Processing results
        self.analysis_data = {}
        
    def run_complete_pipeline(self, target_ratio: float = 1.0) -> bool:
        """
        Run complete processing pipeline with truly sequential processing:
        Load Reference -> Process Scan 1 Completely -> Process Scan 2 Completely -> Visualize
        
        Args:
            target_ratio: Target scaling ratio (1.0 for 1:1 scaling)
            
        Returns:
            Success status
        """
        print("="*100)
        print("MILL ANALYSIS - TRULY SEQUENTIAL PROCESSING PIPELINE")
        print("V3 Noise Removal + old_version Alignment Algorithms")
        if self.use_file_selector:
            print("With Intelligent File Selection")
        if self.dual_scan_mode:
            print("With Truly Sequential Dual-Scan Mode for Proper Alignment")
        print("="*100)
        
        start_time = time.time()
        
        try:
            # STEP 1: Load reference and get scan paths
            if not self._step_1_load_reference_and_get_scan_paths():
                return False
            
            # STEP 2: Process Scan 1 completely (load -> process -> save -> cleanup)
            if not self._step_2_process_scan_1_completely(target_ratio):
                return False
            
            # STEP 3: Process Scan 2 completely (if dual-scan mode)
            if self.dual_scan_mode:
                if not self._step_3_process_scan_2_completely(target_ratio):
                    return False
                
                # STEP 3.5: Apply post-processing ICP between scans if dual-scan mode
                # This is automatically handled by the updated AlignmentProcessor
                print("\n[STEP 3.5] CHECKING FOR POST-PROCESSING ICP BETWEEN SCANS")
                if self.alignment_processor.apply_post_processing_icp_if_ready():
                    print("SUCCESS: Post-processing ICP applied between scans")
                    
                    # Update our references to the aligned scans
                    dual_scan_results = self.alignment_processor.get_dual_scan_results()
                    if 'scan1_aligned' in dual_scan_results and 'scan2_aligned' in dual_scan_results:
                        self.inner_scan1_aligned = dual_scan_results['scan1_aligned']
                        self.inner_scan2_aligned = dual_scan_results['scan2_aligned']
                        print("SUCCESS: Updated aligned scans with post-ICP results")
            
            # STEP 4: Create final visualizations using visualization.py
            if not self._step_4_create_final_visualizations():
                return False
            
            # STEP 5: Generate final summary
            self._step_5_final_summary(start_time)
            
            return True
            
        except Exception as e:
            print(f"CRITICAL ERROR in pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _step_1_load_reference_and_get_scan_paths(self) -> bool:
        """Step 1: Load reference once and get inner scan file paths."""
        print("\n[STEP 1] LOADING REFERENCE AND GETTING SCAN PATHS")
        if self.use_file_selector:
            print("Using Intelligent File Selection")
        else:
            print("Using Hardcoded File Paths")
        if self.dual_scan_mode:
            print("Getting paths for 2 inner scans")
        print("="*60)
        
        # Initialize file selector first with correct dual_scan_mode if using file selector
        if self.use_file_selector:
            self.data_loader.initialize_file_selector(dual_scan_mode=self.dual_scan_mode)
            
            # Perform intelligent file selection first
            selection_results = self.data_loader.perform_intelligent_file_selection(dual_scan_mode=self.dual_scan_mode)
            
            # Handle both boolean and dictionary return types
            if isinstance(selection_results, bool):
                selection_successful = selection_results
            elif isinstance(selection_results, dict):
                selection_successful = selection_results.get('selection_successful', False)
            else:
                selection_successful = False
            
            if not selection_successful:
                print("ERROR: File selection failed")
                return False
            
            # Get the selected files from data_loader
            if hasattr(self.data_loader, 'selected_files') and self.data_loader.selected_files:
                # Get reference file path
                reference_file_path = self.data_loader.selected_files['reference']
                
                # Load reference using the selected path directly
                self.reference_pcd = self.data_loader.load_point_cloud_file(reference_file_path)
                if self.reference_pcd is None:
                    print("ERROR: Failed to load reference file")
                    return False
                
                ref_analysis = self.data_loader.get_point_cloud_info(self.reference_pcd, "Reference Shell")
                
                # Get inner scan paths
                self.inner_scan1_path = self.data_loader.selected_files['inner_scan']
                if self.dual_scan_mode:
                    self.inner_scan2_path = self.data_loader.selected_files.get('inner_scan2', self.inner_scan1_path)
            else:
                print("ERROR: Selected files not found")
                return False
        else:
            # Use hardcoded paths
            from config import get_file_paths
            file_paths = get_file_paths()
            
            # Load reference using hardcoded path
            self.reference_pcd = self.data_loader.load_point_cloud_file(file_paths['reference'])
            if self.reference_pcd is None:
                print("ERROR: Failed to load reference file")
                return False
            
            ref_analysis = self.data_loader.get_point_cloud_info(self.reference_pcd, "Reference Shell")
            
            # Get inner scan paths
            self.inner_scan1_path = file_paths['inner_scan']
            if self.dual_scan_mode:
                self.inner_scan2_path = file_paths['inner_scan']  # Same file for compatibility
        
        # Create analysis data
        self.analysis_data = {
            'reference': ref_analysis,
            'loading_successful': True,
            'dual_scan_mode': self.dual_scan_mode,
            'intelligent_file_selection_used': self.use_file_selector
        }
        
        print(f"SUCCESS: Reference loaded, scan paths obtained")
        print(f"  Reference: {ref_analysis['num_points']:,} points")
        print(f"  Scan 1 path: {self.inner_scan1_path.name}")
        if self.dual_scan_mode:
            print(f"  Scan 2 path: {self.inner_scan2_path.name}")
        
        return True
    
    def _step_2_process_scan_1_completely(self, target_ratio: float) -> bool:
        """Step 2: Load and process Scan 1 completely, then save and cleanup."""
        print("\n[STEP 2] PROCESSING SCAN 1 COMPLETELY (LOAD -> PROCESS -> SAVE -> CLEANUP)")
        print("="*60)
        
        # Clear any existing scan data
        gc.collect()
        
        # Sub-step 2.1: Load Scan 1
        print("\n[STEP 2.1] LOADING SCAN 1")
        scan1_raw = self.data_loader.load_point_cloud_file(self.inner_scan1_path)
        
        if scan1_raw is None:
            print("ERROR: Failed to load scan 1")
            return False
        
        scan1_analysis = self.data_loader.get_point_cloud_info(scan1_raw, "Inner Scan 1")
        self.analysis_data['inner_scan'] = scan1_analysis
        
        print(f"SUCCESS: Scan 1 loaded - {scan1_analysis['num_points']:,} points")
        
        # Sub-step 2.2: Noise Removal for Scan 1
        print("\n[STEP 2.2] NOISE REMOVAL - SCAN 1")
        scan1_cleaned = self.noise_processor.apply_v3_noise_removal(scan1_raw)
        
        if scan1_cleaned is None:
            print("ERROR: Noise removal failed for scan 1")
            return False
        
        print("SUCCESS: Noise removal completed for scan 1")
        
        # Create noise removal visualization with source filename
        if hasattr(self.visualization_processor, 'create_noise_removal_visualization'):
            try:
                print("Creating noise removal visualization for scan 1...")
                # Extract source filename
                scan1_filename = self.inner_scan1_path.name if hasattr(self.inner_scan1_path, 'name') else str(self.inner_scan1_path)
                self.visualization_processor.create_noise_removal_visualization(
                    scan1_raw, scan1_cleaned, source_filename=scan1_filename
                )
            except Exception as e:
                print(f"WARNING: Noise removal visualization failed: {e}")
        
        # Cleanup raw scan1
        del scan1_raw
        gc.collect()
        
        # Sub-step 2.3 & 2.4: Scaling and Alignment for Scan 1 (using enhanced alignment processor)
        print("\n[STEP 2.3-2.4] SCALING AND ALIGNMENT - SCAN 1")
        
        # Use the enhanced alignment processor with fixed scaling
        self.inner_scan1_aligned, self.scale_factor1 = self.alignment_processor.apply_complete_alignment_with_fixed_scaling(
            self.reference_pcd, scan1_cleaned, scan_identifier="scan1"
        )
        
        if self.inner_scan1_aligned is None:
            print("ERROR: Enhanced alignment failed for scan 1")
            return False
        
        print(f"SUCCESS: Enhanced alignment applied to scan 1 (scale factor: {self.scale_factor1:.6f})")
        
        # Create alignment visualization with source filename
        if hasattr(self.visualization_processor, 'create_alignment_visualization'):
            try:
                print("Creating alignment visualization for scan 1...")
                scan1_filename = self.inner_scan1_path.name if hasattr(self.inner_scan1_path, 'name') else str(self.inner_scan1_path)
                
                # Create a scaled copy for visualization
                scan1_scaled = copy.deepcopy(scan1_cleaned)
                scan1_scaled.scale(self.scale_factor1, center=scan1_scaled.get_center())
                
                self.visualization_processor.create_alignment_visualization(
                    self.reference_pcd, scan1_scaled, self.inner_scan1_aligned, 
                    scan_name="scan1", source_filename=scan1_filename
                )
                
                # Cleanup scaled copy
                del scan1_scaled
            except Exception as e:
                print(f"WARNING: Alignment visualization failed: {e}")
        
        # Cleanup cleaned scan1
        del scan1_cleaned
        gc.collect()
        
        # Save aligned scan 1 to disk for heatmap analysis
        output_dir = Path("output/aligned_scans")
        output_dir.mkdir(parents=True, exist_ok=True)
        scan1_path = output_dir / "scan1_aligned.ply"
        o3d.io.write_point_cloud(str(scan1_path), self.inner_scan1_aligned)
        print(f"Scan 1 aligned result saved to: {scan1_path}")
        
        return True
    
    def _step_3_process_scan_2_completely(self, target_ratio: float) -> bool:
        """Step 3: Load and process Scan 2 completely, then save and cleanup."""
        print("\n[STEP 3] PROCESSING SCAN 2 COMPLETELY (LOAD -> PROCESS -> SAVE -> CLEANUP)")
        print("="*60)
        
        # Clear any temporary data
        gc.collect()
        
        # Sub-step 3.1: Load Scan 2
        print("\n[STEP 3.1] LOADING SCAN 2")
        scan2_raw = self.data_loader.load_point_cloud_file(self.inner_scan2_path)
        
        if scan2_raw is None:
            print("ERROR: Failed to load scan 2")
            return False
        
        scan2_analysis = self.data_loader.get_point_cloud_info(scan2_raw, "Inner Scan 2")
        self.analysis_data['inner_scan2'] = scan2_analysis
        
        print(f"SUCCESS: Scan 2 loaded - {scan2_analysis['num_points']:,} points")
        
        # Sub-step 3.2: Noise Removal for Scan 2
        print("\n[STEP 3.2] NOISE REMOVAL - SCAN 2")
        scan2_cleaned = self.noise_processor.apply_v3_noise_removal(scan2_raw)
        
        if scan2_cleaned is None:
            print("ERROR: Noise removal failed for scan 2")
            return False
        
        print("SUCCESS: Noise removal completed for scan 2")
        
        # Create noise removal visualization with source filename
        if hasattr(self.visualization_processor, 'create_noise_removal_visualization'):
            try:
                print("Creating noise removal visualization for scan 2...")
                scan2_filename = self.inner_scan2_path.name if hasattr(self.inner_scan2_path, 'name') else str(self.inner_scan2_path)
                self.visualization_processor.create_noise_removal_visualization(
                    scan2_raw, scan2_cleaned, source_filename=scan2_filename
                )
            except Exception as e:
                print(f"WARNING: Noise removal visualization failed: {e}")
        
        # Cleanup raw scan2
        del scan2_raw
        gc.collect()
        
        # Sub-step 3.3 & 3.4: Scaling and Alignment for Scan 2 (using enhanced alignment processor)
        print("\n[STEP 3.3-3.4] SCALING AND ALIGNMENT - SCAN 2")
        
        # Use the enhanced alignment processor with fixed scaling
        self.inner_scan2_aligned, self.scale_factor2 = self.alignment_processor.apply_complete_alignment_with_fixed_scaling(
            self.reference_pcd, scan2_cleaned, scan_identifier="scan2"
        )
        
        if self.inner_scan2_aligned is None:
            print("ERROR: Enhanced alignment failed for scan 2")
            return False
        
        print(f"SUCCESS: Enhanced alignment applied to scan 2 (scale factor: {self.scale_factor2:.6f})")
        
        # Create alignment visualization with source filename
        if hasattr(self.visualization_processor, 'create_alignment_visualization'):
            try:
                print("Creating alignment visualization for scan 2...")
                scan2_filename = self.inner_scan2_path.name if hasattr(self.inner_scan2_path, 'name') else str(self.inner_scan2_path)
                
                # Create a scaled copy for visualization
                scan2_scaled = copy.deepcopy(scan2_cleaned)
                scan2_scaled.scale(self.scale_factor2, center=scan2_scaled.get_center())
                
                self.visualization_processor.create_alignment_visualization(
                    self.reference_pcd, scan2_scaled, self.inner_scan2_aligned, 
                    scan_name="scan2", source_filename=scan2_filename
                )
                
                # Cleanup scaled copy
                del scan2_scaled
            except Exception as e:
                print(f"WARNING: Alignment visualization failed: {e}")
        
        # Cleanup cleaned scan2
        del scan2_cleaned
        gc.collect()
        
        # Save aligned scan 2 to disk for heatmap analysis
        output_dir = Path("output/aligned_scans")
        output_dir.mkdir(parents=True, exist_ok=True)
        scan2_path = output_dir / "scan2_aligned.ply"
        o3d.io.write_point_cloud(str(scan2_path), self.inner_scan2_aligned)
        print(f"Scan 2 aligned result saved to: {scan2_path}")
        
        return True
    
    def get_output_visualization_files(self) -> Dict[str, str]:
        """Get list of output visualization files with source filename integration."""
        viz_files = {}
        
        if hasattr(self.inner_scan1_path, 'name'):
            scan1_name = Path(self.inner_scan1_path.name).stem
            # Clean filename for filesystem safety
            clean_name = scan1_name.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
            clean_name = clean_name.replace('_converted', '').replace('converted_', '')
            
            viz_files['scan1_noise_removal'] = f"v3_noise_removal_analysis_{clean_name}.png"
            viz_files['scan1_alignment'] = f"alignment_analysis_scan1_{clean_name}.png"
            viz_files['detailed_overlay'] = f"detailed_alignment_overlay_analysis_{clean_name}.png"
        
        if self.dual_scan_mode and hasattr(self.inner_scan2_path, 'name'):
            scan2_name = Path(self.inner_scan2_path.name).stem
            clean_name = scan2_name.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
            clean_name = clean_name.replace('_converted', '').replace('converted_', '')
            
            viz_files['scan2_noise_removal'] = f"v3_noise_removal_analysis_{clean_name}.png"
            viz_files['scan2_alignment'] = f"alignment_analysis_scan2_{clean_name}.png"
        
        # Standard files that don't change
        viz_files['heatmap_analysis'] = "heatmap_wear_analysis.png"
        viz_files['dual_scan_comparison'] = "dual_scan_comparison_analysis.png"
        
        return viz_files
    
    def _step_4_create_final_visualizations(self) -> bool:
        """Step 4: Create final visualizations using existing visualization.py methods."""
        print("\n[STEP 4] CREATING FINAL VISUALIZATIONS USING VISUALIZATION.PY")
        print("="*60)
        
        try:
            # Create detailed overlay visualization with source filename
            if hasattr(self.visualization_processor, 'create_detailed_overlay_visualization'):
                print("Creating detailed overlay visualization...")
                try:
                    scan1_filename = self.inner_scan1_path.name if hasattr(self.inner_scan1_path, 'name') else str(self.inner_scan1_path)
                    self.visualization_processor.create_detailed_overlay_visualization(
                        self.reference_pcd, self.inner_scan1_aligned, source_filename=scan1_filename
                    )
                    print("SUCCESS: Detailed overlay visualization created")
                except Exception as e:
                    print(f"WARNING: Detailed overlay visualization failed: {e}")
            
            # Create two-scan comparison if in dual-scan mode
            if self.dual_scan_mode and hasattr(self.visualization_processor, 'create_two_scan_comparison_visualization'):
                print("Creating two-scan comparison visualization...")
                try:
                    # Create proper data structure - pass point clouds directly
                    scan_data = {
                        'scan1': self.inner_scan1_aligned,  # Pass point cloud objects directly
                        'scan2': self.inner_scan2_aligned,  # Pass point cloud objects directly
                        'reference': self.reference_pcd
                    }
                    
                    # Create heatmap results with available data
                    heatmap_results = {
                        'heatmap_available': True,
                        'mean_distance': 0.0,  # Placeholder - will be updated by heatmap generator
                        'max_distance': 0.0,   # Placeholder - will be updated by heatmap generator
                        'analysis_complete': True
                    }
                    
                    # Call the two-scan comparison method
                    self.visualization_processor.create_two_scan_comparison_visualization(
                        scan_data, self.reference_pcd, heatmap_results
                    )
                    print("SUCCESS: Two-scan comparison visualization created")
                except Exception as e:
                    print(f"WARNING: Two-scan comparison visualization failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # List available methods for debugging
            available_methods = [method for method in dir(self.visualization_processor) 
                            if not method.startswith('_') and 'visualization' in method.lower()]
            
            print(f"Available visualization methods: {available_methods}")
            
        except Exception as e:
            print(f"WARNING: Final visualization step failed: {e}")
        
        print("SUCCESS: Final visualizations completed using visualization.py")
        return True
    
    def _step_5_final_summary(self, start_time):
        """Step 5: Generate final processing summary."""
        total_time = time.time() - start_time
    
        print("\n" + "="*100)
        print("TRULY SEQUENTIAL PROCESSING PIPELINE COMPLETION SUMMARY")
        print("="*100)
    
        if self.analysis_data.get('loading_successful', False):
            ref_points = self.analysis_data['reference']['num_points']
            scan1_points = self.analysis_data['inner_scan']['num_points']
    
            print(f"Data Loading:")
            print(f"  Reference: {ref_points:,} points")
            print(f"  Inner scan 1: {scan1_points:,} points")
    
            if self.dual_scan_mode and 'inner_scan2' in self.analysis_data:
                scan2_points = self.analysis_data['inner_scan2']['num_points']
                print(f"  Inner scan 2: {scan2_points:,} points")
                print(f"  Mode: Truly sequential dual-scan processing")
            else:
                print(f"  Mode: Single-scan processing")
    
            if self.use_file_selector:
                print(f"  File selection: Intelligent selection used")
            else:
                print(f"  File selection: Hardcoded paths used")
    
        print(f"Processing Results:")
        print(f"  Scan 1 scale factor: {self.scale_factor1:.6f}")
        if self.dual_scan_mode:
            print(f"  Scan 2 scale factor: {self.scale_factor2:.6f}")
        print(f"  Target ratio: 1:1")
        print(f"  Fixed scaling enabled: {self.alignment_processor.enable_fixed_scaling}")
    
        # Check alignment quality
        if self.inner_scan1_aligned and self.reference_pcd:
            ref_center = self.reference_pcd.get_center()
            scan1_center = self.inner_scan1_aligned.get_center()
            distance1 = np.linalg.norm(scan1_center - ref_center)
            print(f"  Scan 1 center distance: {distance1:.2f}mm")
    
            if self.dual_scan_mode and self.inner_scan2_aligned:
                scan2_center = self.inner_scan2_aligned.get_center()
                distance2 = np.linalg.norm(scan2_center - ref_center)
                print(f"  Scan 2 center distance: {distance2:.2f}mm")
                
                # If post-processing ICP was applied, show information
                if self.alignment_processor.post_icp_results:
                    print(f"  Post-processing ICP applied between scans: YES")
                    icp_results = self.alignment_processor.post_icp_results
                    print(f"  ICP improvement: {icp_results.get('improvement_mm', 0):.1f}mm")
                    print(f"  ICP quality: {icp_results.get('quality', 'Unknown')}")
                    print(f"  Final scan separation: {icp_results.get('final_separation', 0):.1f}mm")
    
        print(f"Processing:")
        print(f"  Total time: {total_time:.1f} seconds")
        print(f"  Processing approach: Truly sequential (load->process->save->cleanup)")
        print(f"  Memory management: Aggressive cleanup between scans")
        print(f"  Enhanced alignment: Fixed scaling and auto post-processing ICP")
        print(f"  Status: SUCCESSFUL")
    
        print(f"\nVisualization Output:")
        if hasattr(self.inner_scan1_path, 'name'):
            scan1_name = self.inner_scan1_path.name
            print(f"  Scan 1 source: {scan1_name}")
            print(f"  Noise removal viz: v3_noise_removal_analysis_{Path(scan1_name).stem}.png")
            print(f"  Alignment viz: alignment_analysis_scan1_{Path(scan1_name).stem}.png")
            print(f"  Detailed overlay viz: detailed_alignment_overlay_analysis_{Path(scan1_name).stem}.png")
    
        if self.dual_scan_mode and hasattr(self.inner_scan2_path, 'name'):
            scan2_name = self.inner_scan2_path.name
            print(f"  Scan 2 source: {scan2_name}")
            print(f"  Alignment viz: alignment_analysis_scan2_{Path(scan2_name).stem}.png")
    
        print(f"\nReady for Heatmap Analysis:")
        if self.dual_scan_mode:
            print(f"  Scan 1 aligned: {'YES' if self.inner_scan1_aligned else 'NO'}")
            print(f"  Scan 2 aligned: {'YES' if self.inner_scan2_aligned else 'NO'}")
            print(f"  Ready for comparison: {'YES' if (self.inner_scan1_aligned and self.inner_scan2_aligned) else 'NO'}")
            print(f"  Post-ICP alignment: {'APPLIED' if self.alignment_processor.post_icp_results else 'NOT APPLIED'}")
        else:
            print(f"  Single scan aligned: {'YES' if self.inner_scan1_aligned else 'NO'}")
    
        print("="*100)
    
    def run_dual_scan_heatmap_analysis(self, target_ratio: float = 1.0, 
                                     heatmap_type: str = 'enhanced') -> Optional[Dict[str, Any]]:
        """
        Run complete truly sequential pipeline with dual-scan heatmap generation.
        Enhanced with source filename integration in visualizations.
        
        Args:
            target_ratio: Target scaling ratio
            heatmap_type: Type of heatmap ('standard' or 'enhanced')
            
        Returns:
            Combined results dictionary or None if failed
        """
        # First run the truly sequential dual-scan pipeline
        pipeline_success = self.run_complete_pipeline(target_ratio)
        
        if not pipeline_success:
            return None
        
        # Import heatmap generator
        try:
            heatmap_generator = WearHeatmapGenerator()
            print("Generating enhanced heatmap comparison between two sequentially processed scans")
            
            if self.dual_scan_mode and self.inner_scan1_aligned and self.inner_scan2_aligned:
                scan1 = self.inner_scan1_aligned
                scan2 = self.inner_scan2_aligned
                print("Generating heatmap comparison between two sequentially processed scans")
            else:
                # Fallback to same scan comparison
                scan1 = self.inner_scan1_aligned
                scan2 = self.inner_scan1_aligned
                print("WARNING: Single scan mode - generating self-comparison heatmap")
            
            # Call the enhanced heatmap generator
            heatmap_result = heatmap_generator.generate_enhanced_heatmap(scan1, scan2)
            heatmap_generated = True
            
            print("SUCCESS: Enhanced heatmap analysis completed!")
            
        except Exception as e:
            print(f"ERROR: Heatmap generation failed: {e}")
            heatmap_result = None
            heatmap_generated = False

        # Also use visualization.py for heatmap if available
        if hasattr(self.visualization_processor, 'create_heatmap_visualization'):
            try:
                print("Creating heatmap visualization using visualization.py...")
                
                # Get heatmap analysis from generator
                heatmap_analysis = {}
                if hasattr(heatmap_generator, 'heatmap_analysis'):
                    heatmap_analysis = heatmap_generator.heatmap_analysis
                
                # Create combined data structure with proper heatmap availability flag
                combined_heatmap_data = {
                    'scan1': scan1,  # Point cloud objects
                    'scan2': scan2,  # Point cloud objects
                    'reference': self.reference_pcd,
                    'points_scan1': len(np.asarray(scan1.points)),
                    'points_scan2': len(np.asarray(scan2.points)),
                    # Include heatmap analysis results with fallbacks
                    'mean_distance': heatmap_analysis.get('mean_distance', 0),
                    'max_distance': heatmap_analysis.get('max_distance', 0),
                    'min_distance': heatmap_analysis.get('min_distance', 0),
                    'std_distance': heatmap_analysis.get('std_distance', 0),
                    'points_analyzed': heatmap_analysis.get('points_analyzed', 0),
                    'percentile_95': heatmap_analysis.get('percentile_95', 0),
                    'heatmap_type': heatmap_type,
                    'analysis_successful': heatmap_generated,
                    'heatmap_available': heatmap_generated  # Set based on actual success
                }
                
                # Call with single combined parameter
                self.visualization_processor.create_heatmap_visualization(combined_heatmap_data)
                print("SUCCESS: Heatmap visualization created using visualization.py")
                
            except Exception as e:
                print(f"WARNING: visualization.py heatmap failed: {e}")
                import traceback
                traceback.print_exc()

        print("SUCCESS: Truly sequential dual-scan heatmap analysis completed!")

        # Get post-processing ICP results if available
        post_icp_info = None
        if self.alignment_processor.post_icp_results:
            post_icp_info = self.alignment_processor.post_icp_results

        # Combine results with proper heatmap status
        return {
            'processing_successful': True,
            'heatmap_included': heatmap_generated,  # Based on actual result
            'truly_sequential_processing': True,
            'dual_scan_mode': self.dual_scan_mode,
            'pipeline_data': self.analysis_data,
            'heatmap_results': {
                'type': heatmap_type,
                'visualization_created': heatmap_generated,
                'analysis': heatmap_analysis
            },
            'scale_factors': {
                'scan1': self.scale_factor1,
                'scan2': self.scale_factor2 if self.dual_scan_mode else None
            },
            'post_processing_icp': post_icp_info,  # Add post-processing ICP results
            'visualization_files': self.get_output_visualization_files()
        }
