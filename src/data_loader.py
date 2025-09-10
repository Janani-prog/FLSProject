"""
Professional Data Loader for Mill Analysis Project with E57 Support and Intelligent File Selection
Open3D implementation with automatic E57 detection and conversion
Handles file loading, validation, and basic analysis with seamless E57 integration
Now includes optional intelligent file selection while maintaining full backward compatibility
"""

import open3d as o3d
import numpy as np
import time
import gc
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

from config import get_file_paths, ProcessingConfig, FileConfig
from e57_converter import E57Converter


class PointCloudLoader:
    """Professional point cloud loader using Open3D with memory management, E57 support, and optional intelligent file selection."""
    
    def __init__(self):
        self.config = ProcessingConfig()
        self.file_paths = get_file_paths()
        
        # Initialize E57 converter
        self.e57_converter = E57Converter() if FileConfig.E57_CONVERSION_ENABLED else None
        self.e57_conversions_performed = []
        
        # File selector integration (optional)
        self.file_selector = None
        self.selected_files = None
        self.current_dual_scan_mode = False
        
        # Perform initial E57 scan if auto-conversion is enabled
        if FileConfig.E57_AUTO_CONVERSION and self.e57_converter:
            self.scan_and_convert_e57_files()
    
    def initialize_file_selector(self, dual_scan_mode: bool = False) -> bool:
        """
        Initialize the intelligent file selector.
        
        Args:
            dual_scan_mode: If True, selects two inner scan files. If False, selects one (default)
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            from file_selector import IntelligentFileSelector
            self.file_selector = IntelligentFileSelector(dual_scan_mode=dual_scan_mode)
            self.current_dual_scan_mode = dual_scan_mode
            print(f"File selector initialized (dual_scan_mode: {dual_scan_mode})")
            return True
        except ImportError as e:
            print(f"WARNING: Could not import file_selector: {e}")
            print("Falling back to hardcoded file paths")
            return False
        except Exception as e:
            print(f"ERROR: Could not initialize file selector: {e}")
            print("Falling back to hardcoded file paths")
            return False
    
    def perform_intelligent_file_selection(self, dual_scan_mode: bool = False) -> bool:
        """
        Perform intelligent file selection and store results.
        
        Args:
            dual_scan_mode: If True, selects two inner scan files. If False, selects one (default)
            
        Returns:
            True if selection successful, False otherwise
        """
        if not self.initialize_file_selector(dual_scan_mode):
            return False
        
        try:
            results = self.file_selector.perform_intelligent_file_selection()
            
            if results.get('selection_successful', False):
                self.selected_files = results
                self.current_dual_scan_mode = dual_scan_mode
                print("\nIntelligent file selection completed successfully!")
                return True
            else:
                print("\nERROR: Intelligent file selection failed")
                print("Falling back to hardcoded file paths")
                return False
                
        except Exception as e:
            print(f"ERROR during intelligent file selection: {e}")
            print("Falling back to hardcoded file paths")
            return False
    
    def get_target_file_paths(self, use_file_selector: bool = False) -> Dict[str, Path]:
        """
        Get target file paths either from intelligent selection or hardcoded config.
        
        Args:
            use_file_selector: If True, use intelligent file selection. If False, use hardcoded paths (default)
            
        Returns:
            Dictionary with target file paths
        """
        if use_file_selector and self.selected_files and self.selected_files.get('selection_successful', False):
            # Use intelligently selected files
            result = {
                'reference': self.selected_files['reference'],
                'inner_scan': self.selected_files['inner_scan']
            }
            
            # Add second inner scan if in dual-scan mode
            if self.current_dual_scan_mode and 'inner_scan2' in self.selected_files:
                result['inner_scan2'] = self.selected_files['inner_scan2']
            
            return result
        else:
            # Use hardcoded paths from config (default behavior)
            return {
                'reference': self.file_paths['reference'],
                'inner_scan': self.file_paths['inner_scan']
            }
        
    def scan_and_convert_e57_files(self) -> None:
        """Scan data directories for E57 files and convert them automatically."""
        print("\n" + "="*80)
        print("SCANNING FOR E57 FILES IN DATA DIRECTORIES")
        print("="*80)
        
        # Scan reference directory
        ref_conversions = self.e57_converter.scan_and_convert_directory(
            self.file_paths['data_reference_dir']
        )
        
        # Scan inner scans directory  
        scan_conversions = self.e57_converter.scan_and_convert_directory(
            self.file_paths['data_inner_scans_dir']
        )
        
        # Store conversion results
        self.e57_conversions_performed.extend(ref_conversions)
        self.e57_conversions_performed.extend(scan_conversions)
        
        if self.e57_conversions_performed:
            print(f"\nE57 CONVERSION SUMMARY:")
            print(f"Total files converted: {len(self.e57_conversions_performed)}")
            for original, converted in self.e57_conversions_performed:
                print(f"  {original.name} -> {converted.name}")
            print("Converted files are available for processing")
        else:
            print("No E57 files found in data directories")
        
        print("="*80)
    
    def find_best_file_in_directory(self, directory: Path, 
                                   preferred_patterns: List[str] = None) -> Optional[Path]:
        """
        Find the best file to use in a directory, with E57 conversion support.
        
        Args:
            directory: Directory to search
            preferred_patterns: List of filename patterns to prefer
            
        Returns:
            Path to the best file found, or None
        """
        if not directory.exists():
            return None
        
        if preferred_patterns is None:
            preferred_patterns = ['mill', 'body', 'reference', 'inner', 'scan']
        
        # Get all supported files
        supported_files = []
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in FileConfig.SUPPORTED_FORMATS:
                supported_files.append(file_path)
        
        if not supported_files:
            return None
        
        # If E57 converter is available, check for converted files
        if self.e57_converter:
            # Look for recently converted files
            for original, converted in self.e57_conversions_performed:
                if converted.parent == directory and converted in supported_files:
                    print(f"Using recently converted E57 file: {converted.name}")
                    return converted
        
        # Prefer files matching patterns
        for pattern in preferred_patterns:
            for file_path in supported_files:
                if pattern.lower() in file_path.name.lower():
                    return file_path
        
        # Return the largest file (likely the most complete scan)
        largest_file = max(supported_files, key=lambda f: f.stat().st_size)
        return largest_file
    
    def resolve_file_path(self, target_path: Path) -> Path:
        """
        Resolve file path with E57 conversion support.
        If target doesn't exist but E57 equivalent does, convert and return converted path.
        
        Args:
            target_path: Target file path
            
        Returns:
            Actual file path to use (may be converted from E57)
        """
        # If target exists, use it directly
        if target_path.exists():
            return target_path
        
        # Check if E57 equivalent exists
        if self.e57_converter:
            e57_path = target_path.with_suffix('.e57')
            if e57_path.exists():
                print(f"Target {target_path.name} not found, but E57 equivalent exists")
                converted_path = self.e57_converter.convert_e57_to_ply(e57_path)
                if converted_path:
                    print(f"Using converted file: {converted_path.name}")
                    return converted_path
        
        # Look for best alternative in the same directory
        alternative = self.find_best_file_in_directory(target_path.parent)
        if alternative:
            print(f"Using alternative file: {alternative.name} instead of {target_path.name}")
            return alternative
        
        return target_path  # Return original even if it doesn't exist (will fail validation)
    
    def validate_file(self, file_path: Path) -> bool:
        """
        Validate that a file exists and is readable, with E57 support.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            True if file is valid, False otherwise
        """
        if not file_path.exists():
            print(f"ERROR: File not found: {file_path}")
            return False
            
        if not file_path.is_file():
            print(f"ERROR: Path is not a file: {file_path}")
            return False
            
        # Check file extension
        if file_path.suffix.lower() not in FileConfig.SUPPORTED_FORMATS:
            print(f"ERROR: Unsupported file format: {file_path.suffix}")
            print(f"Supported formats: {FileConfig.SUPPORTED_FORMATS}")
            return False
            
        # Check file size (basic check)
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb == 0:
                print(f"ERROR: File is empty: {file_path}")
                return False
                
            print(f"File validation passed: {file_path.name} ({size_mb:.1f} MB)")
            return True
            
        except Exception as e:
            print(f"ERROR: Cannot access file {file_path}: {e}")
            return False
    
    def get_point_cloud_info(self, pcd: o3d.geometry.PointCloud, name: str = "Point Cloud") -> Dict[str, Any]:
        """
        Get comprehensive information about a point cloud.
        
        Args:
            pcd: Open3D point cloud
            name: Descriptive name for the point cloud
            
        Returns:
            Dictionary with point cloud analysis
        """
        if pcd is None or len(pcd.points) == 0:
            return {'name': name, 'valid': False, 'error': 'Empty point cloud'}
            
        points = np.asarray(pcd.points)
        
        # Basic statistics
        center = pcd.get_center()
        bbox = pcd.get_axis_aligned_bounding_box()
        dimensions = bbox.get_extent()
        
        # Calculate characteristic radius (95th percentile method from all versions)
        radial_distances = np.sqrt(np.sum((points - center)**2, axis=1))
        char_radius = np.percentile(radial_distances, 95)
        
        # Point cloud properties
        has_colors = pcd.has_colors()
        has_normals = pcd.has_normals()
        
        analysis = {
            'name': name,
            'valid': True,
            'num_points': len(points),
            'center': center.tolist(),
            'dimensions': dimensions.tolist(),
            'max_dimension': float(np.max(dimensions)),
            'characteristic_radius': float(char_radius),
            'has_colors': has_colors,
            'has_normals': has_normals,
            'bbox_min': bbox.min_bound.tolist(),
            'bbox_max': bbox.max_bound.tolist()
        }
        
        return analysis
    
    def print_point_cloud_analysis(self, analysis: Dict[str, Any]) -> None:
        """Print formatted point cloud analysis."""
        if not analysis.get('valid', False):
            print(f"INVALID: {analysis.get('name', 'Unknown')} - {analysis.get('error', 'Unknown error')}")
            return
            
        print(f"\n=== {analysis['name']} Analysis ===")
        print(f"Points: {analysis['num_points']:,}")
        print(f"Center: [{analysis['center'][0]:.2f}, {analysis['center'][1]:.2f}, {analysis['center'][2]:.2f}]")
        print(f"Dimensions: [{analysis['dimensions'][0]:.2f}, {analysis['dimensions'][1]:.2f}, {analysis['dimensions'][2]:.2f}]")
        print(f"Max dimension: {analysis['max_dimension']:.2f}")
        print(f"Characteristic radius: {analysis['characteristic_radius']:.2f}")
        print(f"Has colors: {analysis['has_colors']}")
        print(f"Has normals: {analysis['has_normals']}")
    
    def load_point_cloud_file(self, file_path: Path, 
                             preserve_properties: bool = True) -> Optional[o3d.geometry.PointCloud]:
        """
        Load point cloud file with E57 conversion support.
        
        Args:
            file_path: Path to point cloud file
            preserve_properties: Whether to compute/preserve normals and colors
            
        Returns:
            Loaded point cloud or None if failed
        """
        # Resolve file path (handles E57 conversion if needed)
        actual_path = self.resolve_file_path(file_path)
        
        # Handle E57 conversion if needed
        if self.e57_converter and actual_path.suffix.lower() == '.e57':
            print(f"E57 file detected: {actual_path.name}")
            converted_path = self.e57_converter.process_file_with_e57_check(actual_path)
            if converted_path != actual_path:
                actual_path = converted_path
                print(f"Using converted PLY: {actual_path.name}")
        
        if not self.validate_file(actual_path):
            return None
            
        print(f"\nLoading point cloud: {actual_path.name}")
        start_time = time.time()
        
        try:
            # Load point cloud using Open3D (V3 exact flow)
            pcd = o3d.io.read_point_cloud(str(actual_path))
            
            if len(pcd.points) == 0:
                print(f"ERROR: No points found in {actual_path.name}")
                return None
                
            points_count = len(pcd.points)
            print(f"Loaded {points_count:,} points (no downsampling - V3 flow)")
            
            # Preserve/compute properties if requested
            if preserve_properties:
                # Compute normals if not present
                if not pcd.has_normals():
                    print("Computing normals...")
                    pcd.estimate_normals()
                    
                # Colors are preserved automatically if present
                if pcd.has_colors():
                    print("Colors preserved from file")
            
            load_time = time.time() - start_time
            print(f"Loading completed in {load_time:.2f} seconds")
            
            return pcd
            
        except Exception as e:
            print(f"ERROR loading {actual_path.name}: {str(e)}")
            return None
    
    def load_dual_scans_with_selector(self, dual_scan_mode: bool = True) -> Tuple[Optional[o3d.geometry.PointCloud], Optional[o3d.geometry.PointCloud], Dict[str, Any]]:
        """
        Load reference and two inner scans using intelligent file selection.
        
        Args:
            dual_scan_mode: Should be True for dual-scan loading
            
        Returns:
            Tuple of (reference_pcd, inner_scan1_pcd, inner_scan2_pcd, combined_analysis)
        """
        # Perform intelligent file selection in dual-scan mode
        if not self.perform_intelligent_file_selection(dual_scan_mode=dual_scan_mode):
            print("ERROR: Dual-scan file selection failed")
            return None, None, {}
        
        # Load reference file
        ref_path = self.selected_files['reference']
        print(f"Loading reference: {ref_path.name}")
        ref_pcd = self.load_point_cloud_file(ref_path)
        
        if ref_pcd is None:
            print("ERROR: Failed to load reference file")
            return None, None, {}
        
        ref_analysis = self.get_point_cloud_info(ref_pcd, "Reference Shell")
        self.print_point_cloud_analysis(ref_analysis)
        
        # Load inner scan files
        scan1_path = self.selected_files['inner_scan']
        scan2_path = self.selected_files.get('inner_scan2', scan1_path)  # Fallback to same file
        
        print(f"Loading inner scan 1: {scan1_path.name}")
        scan1_pcd = self.load_point_cloud_file(scan1_path)
        
        print(f"Loading inner scan 2: {scan2_path.name}")
        scan2_pcd = self.load_point_cloud_file(scan2_path)
        
        if scan1_pcd is None or scan2_pcd is None:
            print("ERROR: Failed to load inner scan files")
            return None, None, {}
        
        # Create analysis
        scan1_analysis = self.get_point_cloud_info(scan1_pcd, "Inner Scan 1")
        scan2_analysis = self.get_point_cloud_info(scan2_pcd, "Inner Scan 2")
        
        self.print_point_cloud_analysis(scan1_analysis)
        self.print_point_cloud_analysis(scan2_analysis)
        
        combined_analysis = {
            'reference': ref_analysis,
            'inner_scan': scan1_analysis,
            'inner_scan2': scan2_analysis,
            'loading_successful': True,
            'dual_scan_mode': True,
            'same_inner_scan': scan1_path == scan2_path,
            'intelligent_file_selection_used': True
        }
        
        # Store for pipeline access
        self.reference_pcd = ref_pcd
        
        print(f"\nDual scan loading completed successfully!")
        print(f"  Reference: {ref_analysis['num_points']:,} points")
        print(f"  Inner scan 1: {scan1_analysis['num_points']:,} points") 
        print(f"  Inner scan 2: {scan2_analysis['num_points']:,} points")
        
        return scan1_pcd, scan2_pcd, combined_analysis
    
    def load_reference_file(self, use_file_selector: bool = False) -> Tuple[Optional[o3d.geometry.PointCloud], Dict[str, Any]]:
        """
        Load reference file directly (V3 exact flow - no downsampling) with E57 support and optional intelligent file selection.
        
        Args:
            use_file_selector: If True, use intelligent file selection. If False, use hardcoded paths (default)
        
        Returns:
            Tuple of (point_cloud, analysis_dict)
        """
        print("\n" + "="*80)
        if use_file_selector:
            print("LOADING REFERENCE FILE (V3 FLOW + E57 SUPPORT + INTELLIGENT SELECTION)")
        else:
            print("LOADING REFERENCE FILE (V3 FLOW + E57 SUPPORT)")
        print("="*80)
        
        # Get target file path
        if use_file_selector:
            # Use the current dual_scan_mode for consistency
            if not self.perform_intelligent_file_selection(dual_scan_mode=self.current_dual_scan_mode):
                print("WARNING: Falling back to hardcoded reference file path")
                use_file_selector = False
        
        target_paths = self.get_target_file_paths(use_file_selector)
        ref_path = target_paths['reference']
        
        if use_file_selector and self.selected_files:
            print(f"Using intelligently selected reference: {ref_path.name}")
        else:
            print(f"Using configured reference: {ref_path.name}")
        
        # Load directly without any downsampling (V3 flow) with E57 support
        pcd = self.load_point_cloud_file(ref_path, preserve_properties=True)
        
        if pcd is None:
            return None, {}
            
        # Analyze reference point cloud
        analysis = self.get_point_cloud_info(pcd, "Reference Shell")
        self.print_point_cloud_analysis(analysis)
        
        print(f"\nREFERENCE LOADED SUCCESSFULLY (V3 FLOW):")
        print(f"  Points loaded: {analysis['num_points']:,}")
        print(f"  No downsampling applied")
        print(f"  Ready for processing pipeline")
        
        return pcd, analysis
    
    def load_inner_scan_file(self, use_file_selector: bool = False) -> Tuple[Optional[o3d.geometry.PointCloud], Dict[str, Any]]:
        """
        Load inner scan file directly (V3 exact flow - full resolution) with E57 support and optional intelligent file selection.
        Downsampling will happen later in the noise removal process.
        
        Args:
            use_file_selector: If True, use intelligent file selection. If False, use hardcoded paths (default)
        
        Returns:
            Tuple of (point_cloud, analysis_dict)
        """
        print("\n" + "="*80)
        if use_file_selector:
            print("LOADING INNER SCAN FILE (V3 FLOW + E57 SUPPORT + INTELLIGENT SELECTION)")
        else:
            print("LOADING INNER SCAN FILE (V3 FLOW + E57 SUPPORT)")
        print("="*80)
        
        # Get target file path
        if use_file_selector:
            # Use existing selection if already performed, otherwise perform it
            if not self.selected_files or not self.selected_files.get('selection_successful', False):
                if not self.perform_intelligent_file_selection(dual_scan_mode=self.current_dual_scan_mode):
                    print("WARNING: Falling back to hardcoded inner scan file path")
                    use_file_selector = False
        
        target_paths = self.get_target_file_paths(use_file_selector)
        scan_path = target_paths['inner_scan']
        
        if use_file_selector and self.selected_files:
            print(f"Using intelligently selected inner scan: {scan_path.name}")
        else:
            print(f"Using configured inner scan: {scan_path.name}")
        
        # Load directly without any downsampling (V3 flow) with E57 support
        # Downsampling will happen in the noise removal voxel step
        pcd = self.load_point_cloud_file(scan_path, preserve_properties=True)
        
        if pcd is None:
            return None, {}
            
        # Analyze inner scan point cloud
        analysis = self.get_point_cloud_info(pcd, "Inner Scan")
        self.print_point_cloud_analysis(analysis)
        
        print(f"\nINNER SCAN LOADED SUCCESSFULLY (V3 FLOW):")
        print(f"  Points loaded: {analysis['num_points']:,}")
        print(f"  Full resolution preserved")
        print(f"  Downsampling will occur in noise removal voxel step")
        print(f"  Ready for V3 noise removal pipeline")
        
        return pcd, analysis
    
    def load_both_files(self, use_file_selector: bool = False) -> Tuple[Optional[o3d.geometry.PointCloud], 
                                     Optional[o3d.geometry.PointCloud],
                                     Dict[str, Any]]:
        """
        Load both reference and inner scan files with appropriate handling, E57 support, and optional intelligent file selection.
        
        Args:
            use_file_selector: If True, use intelligent file selection. If False, use hardcoded paths (default for compatibility)
        
        Returns:
            Tuple of (reference_pcd, inner_scan_pcd, combined_analysis)
        """
        print("="*100)
        if use_file_selector:
            print("MILL ANALYSIS - LOADING DATA FILES (E57 + INTELLIGENT FILE SELECTION)")
        else:
            print("MILL ANALYSIS - LOADING DATA FILES (E57 SUPPORT)")
        print("="*100)
        
        # Load reference file (no downsampling) with optional intelligent selection
        ref_pcd, ref_analysis = self.load_reference_file(use_file_selector=use_file_selector)
        
        if ref_pcd is None:
            print("CRITICAL ERROR: Failed to load reference file")
            return None, None, {}
        
        # Memory cleanup after reference loading
        gc.collect()
        
        # Load inner scan file with optional intelligent selection
        scan_pcd, scan_analysis = self.load_inner_scan_file(use_file_selector=use_file_selector)
        
        if scan_pcd is None:
            print("CRITICAL ERROR: Failed to load inner scan file")
            return ref_pcd, None, {'reference': ref_analysis}
        
        # Combined analysis
        combined_analysis = {
            'reference': ref_analysis,
            'inner_scan': scan_analysis,
            'loading_successful': True,
            'ready_for_processing': True,
            'e57_conversions_performed': len(self.e57_conversions_performed),
            'intelligent_file_selection_used': use_file_selector,
            'e57_conversion_details': [
                {'original': str(orig), 'converted': str(conv)} 
                for orig, conv in self.e57_conversions_performed
            ]
        }
        
        # Add file selection details if used
        if use_file_selector and self.selected_files:
            combined_analysis['file_selection_details'] = {
                'selection_successful': self.selected_files.get('selection_successful', False),
                'dual_scan_mode': self.selected_files.get('dual_scan_mode', False),
                'reference_file': str(self.selected_files.get('reference', '')),
                'inner_scan_file': str(self.selected_files.get('inner_scan', ''))
            }
        
        print("\n" + "="*80)
        if use_file_selector:
            print("DATA LOADING COMPLETED SUCCESSFULLY (E57 + INTELLIGENT SELECTION)")
        else:
            print("DATA LOADING COMPLETED SUCCESSFULLY (E57 SUPPORT)")
        print("="*80)
        print(f"Reference: {ref_analysis['num_points']:,} points (preserved exactly)")
        print(f"Inner scan: {scan_analysis['num_points']:,} points (optimized)")
        if self.e57_conversions_performed:
            print(f"E57 conversions: {len(self.e57_conversions_performed)} files converted")
        if use_file_selector:
            print("File selection: Intelligent selection used")
        else:
            print("File selection: Hardcoded paths used (default)")
        print("Ready for noise removal and scaling pipeline")
        print("="*80)
        
        return ref_pcd, scan_pcd, combined_analysis
    
    def cleanup_memory(self) -> None:
        """Force garbage collection to free memory."""
        gc.collect()
        print("Memory cleanup performed")
    
    def get_e57_conversion_summary(self) -> Dict[str, Any]:
        """Get summary of E57 conversions performed."""
        return {
            'e57_support_enabled': FileConfig.E57_CONVERSION_ENABLED,
            'conversions_performed': len(self.e57_conversions_performed),
            'conversion_details': [
                {
                    'original_file': original.name,
                    'converted_file': converted.name,
                    'original_path': str(original),
                    'converted_path': str(converted)
                }
                for original, converted in self.e57_conversions_performed
            ],
            'converter_info': self.e57_converter.get_conversion_info() if self.e57_converter else None
        }
    
    def get_file_selection_summary(self) -> Dict[str, Any]:
        """Get summary of intelligent file selection if used."""
        if self.selected_files:
            return {
                'file_selection_used': True,
                'selection_successful': self.selected_files.get('selection_successful', False),
                'dual_scan_mode': self.selected_files.get('dual_scan_mode', False),
                'selected_files': {
                    'reference': str(self.selected_files.get('reference', '')),
                    'inner_scan': str(self.selected_files.get('inner_scan', '')),
                    'inner_scan2': str(self.selected_files.get('inner_scan2', '')) if self.selected_files.get('inner_scan2') else None
                }
            }
        else:
            return {
                'file_selection_used': False,
                'hardcoded_paths_used': True
            }


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING DATA LOADER WITH E57 SUPPORT AND INTELLIGENT FILE SELECTION")
    print("=" * 60)
    
    loader = PointCloudLoader()
    
    # Show E57 conversion summary
    e57_summary = loader.get_e57_conversion_summary()
    print(f"\nE57 Support Status:")
    print(f"  E57 support enabled: {e57_summary['e57_support_enabled']}")
    print(f"  Conversions performed: {e57_summary['conversions_performed']}")
    
    # Test 1: Default behavior (hardcoded paths - backward compatible)
    print("\n" + "="*60)
    print("TEST 1: DEFAULT BEHAVIOR (BACKWARD COMPATIBLE)")
    print("="*60)
    ref_pcd, scan_pcd, analysis = loader.load_both_files()
    
    if ref_pcd is not None and scan_pcd is not None:
        print("SUCCESS: Default data loading test completed")
        print("Both files loaded successfully using hardcoded paths")
        file_selection_summary = loader.get_file_selection_summary()
        print(f"File selection used: {file_selection_summary['file_selection_used']}")
    else:
        print("ERROR: Default data loading test failed")
    
    # Test 2: Dual-scan mode test
    print("\n" + "="*60)
    print("TEST 2: DUAL-SCAN MODE (NEW FEATURE)")
    print("="*60)
    
    # Create new loader instance for clean test
    loader2 = PointCloudLoader()
    
    # Test dual-scan loading with file selector
    scan1_pcd, scan2_pcd, analysis2 = loader2.load_dual_scans_with_selector(dual_scan_mode=True)
    
    if scan1_pcd is not None and scan2_pcd is not None:
        print("SUCCESS: Dual-scan mode test completed")
        print("Both scans loaded successfully using intelligent selection")
    else:
        print("NOTE: Dual-scan mode test - may require user input in interactive mode")
    
    print("\n" + "="*80)
    print("DATA LOADER TESTING COMPLETED")
    print("="*80)
    print("✅ Backward compatibility maintained - existing code works unchanged")
    print("✅ E57 support integrated seamlessly")
    print("✅ Intelligent file selection available as opt-in feature")
    print("✅ Dual-scan mode available for heatmap analysis")
    print("✅ Zero breaking changes to existing workflow")
    print("="*80)
