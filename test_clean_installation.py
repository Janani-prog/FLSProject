"""
Clean Open3D Installation Test for Mill Analysis Project
Verify that Open3D works correctly with actual data files
"""

import open3d as o3d
import numpy as np
from pathlib import Path
import sys
import os

def test_open3d_installation():
    """Test Open3D installation and basic functionality."""
    print("=" * 80)
    print("TESTING CLEAN OPEN3D INSTALLATION")
    print("=" * 80)
    
    try:
        print(f"Open3D version: {o3d.__version__}")
        print(f"NumPy version: {np.__version__}")
        print(f"Python version: {sys.version}")
        return True
    except Exception as e:
        print(f"Installation test failed: {e}")
        return False

def test_data_file_access():
    """Test access to actual data files."""
    print("\n" + "=" * 60)
    print("TESTING DATA FILE ACCESS")
    print("=" * 60)
    
    # Get project root - fix the path calculation
    project_root = Path(__file__).parent.absolute()
    print(f"Project root: {project_root}")
    
    # Define file paths
    reference_path = project_root / 'data' / 'reference' / 'Mill body - vale verde - v1 (1).PLY'
    inner_scan_path = project_root / 'data' / 'inner_scans' / '8_8-Aug-2024.ply'
    
    # Test reference file
    print(f"\nReference file: {reference_path.name}")
    print(f"Full path: {reference_path}")
    if reference_path.exists():
        size_mb = reference_path.stat().st_size / (1024 * 1024)
        print(f"SUCCESS: File exists, size: {size_mb:.1f} MB")
    else:
        print(f"ERROR: File not found at {reference_path}")
        return False
    
    # Test inner scan file  
    print(f"\nInner scan file: {inner_scan_path.name}")
    print(f"Full path: {inner_scan_path}")
    if inner_scan_path.exists():
        size_mb = inner_scan_path.stat().st_size / (1024 * 1024)
        print(f"SUCCESS: File exists, size: {size_mb:.1f} MB")
    else:
        print(f"ERROR: File not found at {inner_scan_path}")
        return False
        
    return True

def test_open3d_file_loading():
    """Test Open3D file loading with actual data."""
    print("\n" + "=" * 60)
    print("TESTING OPEN3D FILE LOADING")
    print("=" * 60)
    
    project_root = Path(__file__).parent.absolute()
    reference_path = project_root / 'data' / 'reference' / 'Mill body - vale verde - v1 (1).PLY'
    
    try:
        print("Loading reference file with Open3D...")
        print(f"Loading from: {reference_path}")
        pcd = o3d.io.read_point_cloud(str(reference_path))
        
        if len(pcd.points) == 0:
            print("ERROR: Loaded point cloud is empty")
            return False
            
        points = np.asarray(pcd.points)
        print(f"SUCCESS: Loaded {len(points):,} points")
        
        # Basic analysis
        center = pcd.get_center()
        bbox = pcd.get_axis_aligned_bounding_box()
        dimensions = bbox.get_extent()
        
        print(f"Center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
        print(f"Dimensions: [{dimensions[0]:.2f}, {dimensions[1]:.2f}, {dimensions[2]:.2f}]")
        print(f"Max dimension: {np.max(dimensions):.2f}")
        
        return True
        
    except Exception as e:
        print(f"Open3D loading failed: {e}")
        return False

def test_template_modules():
    """Test that template modules can be imported."""
    print("\n" + "=" * 60)
    print("TESTING TEMPLATE MODULES")
    print("=" * 60)
    
    # Add src to path properly
    project_root = Path(__file__).parent.absolute()
    src_path = project_root / 'src'
    sys.path.insert(0, str(src_path))
    
    print(f"Adding to Python path: {src_path}")
    
    try:
        from config import ProcessingConfig, get_file_paths
        print("SUCCESS: config.py imported")
        
        config = ProcessingConfig()
        paths = get_file_paths()
        print(f"SUCCESS: Configuration loaded, {len(paths)} file paths defined")
        
        # Test a few key paths
        print(f"Reference path configured: {paths['reference']}")
        print(f"Inner scan path configured: {paths['inner_scan']}")
        
    except Exception as e:
        print(f"Template import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

def run_complete_test():
    """Run complete installation and setup test."""
    print("MILL ANALYSIS - CLEAN INSTALLATION TEST")
    print("Testing Open3D + Data Files + Template Modules")
    print()
    
    tests = [
        ("Open3D Installation", test_open3d_installation),
        ("Data File Access", test_data_file_access),
        ("Open3D File Loading", test_open3d_file_loading),
        ("Template Modules", test_template_modules)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{status}: {test_name}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
        print("Clean installation ready for implementation")
        print("Data files accessible")
        print("Open3D working correctly")
        print("Template modules importable")
        print("\nReady to start Phase 1: Data Loading implementation")
    else:
        print("SOME TESTS FAILED")
        print("Please check the errors above and resolve before proceeding")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    run_complete_test()
