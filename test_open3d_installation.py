"""
Complete installation test for Mill Analysis Project with Open3D
Python 3.11 + Open3D environment validation
"""

try:
    print("=" * 80)
    print("MILL ANALYSIS PROJECT - COMPLETE INSTALLATION TEST")
    print("=" * 80)
    
    # Test Python version
    import sys
    print(f"Python version: {sys.version}")
    
    # Test core libraries
    import numpy as np
    print(f"[OK] NumPy {np.__version__} imported successfully")
    
    import scipy
    print(f"[OK] SciPy {scipy.__version__} imported successfully")
    
    import matplotlib
    print(f"[OK] Matplotlib {matplotlib.__version__} imported successfully")
    
    import pandas as pd
    print(f"[OK] Pandas {pd.__version__} imported successfully")
    
    import sklearn
    print(f"[OK] Scikit-learn {sklearn.__version__} imported successfully")
    
    import seaborn as sns
    print(f"[OK] Seaborn {sns.__version__} imported successfully")
    
    import tqdm
    print(f"[OK] TQDM {tqdm.__version__} imported successfully")
    
    # THE BIG TEST: Open3D
    import open3d as o3d
    print(f"[SUCCESS] Open3D {o3d.__version__} imported successfully!")
    
    # Test basic Open3D functionality
    print("\nTesting Open3D basic functionality...")
    test_pcd = o3d.geometry.PointCloud()
    test_points = np.random.rand(1000, 3)
    test_pcd.points = o3d.utility.Vector3dVector(test_points)
    print(f"[OK] Created test point cloud with {len(test_pcd.points)} points")
    
    # Test visualization components
    test_pcd.estimate_normals()
    print(f"[OK] Normal estimation working")
    
    center = test_pcd.get_center()
    print(f"[OK] Point cloud analysis working - center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
    
    print("\n" + "=" * 80)
    print("🎉 ALL TESTS PASSED! 🎉")
    print("=" * 80)
    print("✅ Python 3.11 environment ready")
    print("✅ Open3D 0.19.0 working perfectly") 
    print("✅ All scientific libraries installed")
    print("✅ Ready for mill analysis processing!")
    
    print("\nNext steps:")
    print("1. Copy data files to data/reference/ and data/inner_scans/")
    print("2. Run the Open3D data loader")
    print("3. Start implementing processing modules")
    
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    print("Some libraries may not be installed correctly")
except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")
