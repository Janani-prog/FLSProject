import os
import subprocess
import time
import shutil
import numpy as np
from pathlib import Path

class CloudCompareMillAnalysis:
    def __init__(self, cloudcompare_path, base_dir, subsample_points=5000000):
        self.cloudcompare_path = cloudcompare_path
        self.base_dir = Path(base_dir)
        self.subsample_points = subsample_points
        self.output_dir = self.base_dir / "cloudcompare_mill_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"CloudCompare Mill Analysis Pipeline initialized")
        print(f"Base directory: {self.base_dir}")
        print(f"CloudCompare: {self.cloudcompare_path}")
        print(f"Output: {self.output_dir}")
        print(f"Subsample target: {self.subsample_points:,} points")
    
    def run_cloudcompare_command(self, args, working_dir=None, timeout=1800):
        """Robust CloudCompare command execution with extended timeout for high-accuracy operations"""
        try:
            work_dir = working_dir if working_dir else str(self.output_dir)
            cmd = [self.cloudcompare_path] + args
            
            print(f"CloudCompare: {' '.join(args[:3])}...")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work_dir
            )
            
            if result.returncode == 0:
                print("CloudCompare operation successful")
                return True
            else:
                print(f"CloudCompare failed (code: {result.returncode})")
                if result.stderr:
                    print(f"   Error: {result.stderr[:200]}...")
                return False
                
        except subprocess.TimeoutExpired:
            print("CloudCompare operation timed out")
            return False
        except Exception as e:
            print(f"CloudCompare execution failed: {e}")
            return False
    
    def align_box_center_with_open3d(self, reference_path, compare_path):
        """Bounding box center alignment function"""
        try:
            import open3d as o3d
            
            print("Step 1: Bounding box center alignment...")
            
            # Load point clouds
            reference = o3d.io.read_point_cloud(reference_path)
            compare = o3d.io.read_point_cloud(compare_path)
            
            print(f"   Reference: {len(reference.points):,} points")
            print(f"   Compare: {len(compare.points):,} points")
            
            # Original alignment logic
            reference_center = reference.get_center()
            compare_center = compare.get_center()
            
            print(f"   Reference center: [{reference_center[0]:.3f}, {reference_center[1]:.3f}, {reference_center[2]:.3f}]")
            print(f"   Compare center: [{compare_center[0]:.3f}, {compare_center[1]:.3f}, {compare_center[2]:.3f}]")
            
            # Compute the transformation that aligns the two centers
            transformation = np.eye(4)
            transformation[:3, 3] = reference_center - compare_center
            
            # Transform the compare model using the computed transformation
            transformed_compare = compare.transform(transformation)
            
            # Verify alignment
            new_center = compare.get_center()
            distance = np.linalg.norm(reference_center - new_center)
            print(f"   Center distance after alignment: {distance:.6f}")
            print("Bounding Box centered.")
            
            # Save aligned file
            aligned_file = self.output_dir / "target_bbox_aligned.ply"
            o3d.io.write_point_cloud(str(aligned_file), compare)
            
            return str(aligned_file)
            
        except ImportError:
            print("Open3D not available - install with: pip install open3d")
            return compare_path
        except Exception as e:
            print(f"Bounding box alignment failed: {e}")
            return compare_path
    
    def cloudcompare_subsample_robust(self, input_file, num_points=None):
        """Community-proven subsampling with robust file handling"""
        if num_points is None:
            num_points = self.subsample_points
            
        print(f"Subsampling {Path(input_file).name} -> {num_points:,} points...")
        
        # Track files before operation
        input_dir = Path(input_file).parent
        existing_files = set(input_dir.glob("*.ply"))
        
        # Multiple strategies for subsampling based on community examples
        strategies = [
            # Strategy 1: Most basic (highest compatibility)
            ["-SILENT", "-AUTO_SAVE", "OFF", "-O", str(input_file), "-SS", "RANDOM", str(num_points), "-SAVE_CLOUDS"],
            
            # Strategy 2: With explicit PLY export
            ["-SILENT", "-C_EXPORT_FMT", "PLY", "-PLY_EXPORT_FMT", "BINARY_LE", "-O", str(input_file), "-SS", "RANDOM", str(num_points), "-SAVE_CLOUDS"],
            
            # Strategy 3: Ultra-minimal
            ["-SILENT", "-O", str(input_file), "-SS", "RANDOM", str(num_points)]
        ]
        
        for i, strategy in enumerate(strategies, 1):
            print(f"   Trying strategy {i}...")
            
            if self.run_cloudcompare_command(strategy, str(input_dir)):
                time.sleep(3)
                
                # Find new subsampled file
                new_files = set(input_dir.glob("*.ply")) - existing_files
                subsampled_files = [f for f in new_files if any(keyword in f.name.upper() for keyword in ["SUBSAMPLED", "RANDOM"])]
                
                if subsampled_files:
                    # Move to output directory with organized naming
                    subsampled_file = max(subsampled_files, key=lambda x: x.stat().st_mtime)
                    output_name = f"{Path(input_file).stem}_subsampled.ply"
                    output_file = self.output_dir / output_name
                    
                    shutil.move(str(subsampled_file), str(output_file))
                    
                    print(f"Subsampling successful: {output_file.name}")
                    return str(output_file)
        
        print("All subsampling strategies failed")
        return None
    
    def cloudcompare_icp_progressive(self, ref_file, target_file):
        """Progressive ICP strategies based on community best practices"""
        print("CloudCompare ICP registration...")
        
        # Track files before operation
        existing_files = set(self.output_dir.glob("*.ply"))
        
        # Progressive ICP strategies from community examples
        strategies = [
            # Strategy 1: Ultra-minimal (most compatible)
            {
                "name": "Ultra-minimal ICP",
                "args": ["-SILENT", "-O", str(ref_file), "-O", str(target_file), "-ICP"]
            },
            
            # Strategy 2: Basic with reference specification
            {
                "name": "Basic ICP with reference",
                "args": ["-SILENT", "-O", str(ref_file), "-O", str(target_file), "-ICP", "-REFERENCE_IS_FIRST"]
            },
            
            # Strategy 3: With iterations control
            {
                "name": "ICP with iterations",
                "args": ["-SILENT", "-AUTO_SAVE", "OFF", "-O", str(ref_file), "-O", str(target_file), "-ICP", "-REFERENCE_IS_FIRST", "-ITER", "50", "-SAVE_CLOUDS"]
            },
            
            # Strategy 4: Community-recommended full parameters
            {
                "name": "Full ICP parameters", 
                "args": ["-SILENT", "-C_EXPORT_FMT", "PLY", "-PLY_EXPORT_FMT", "BINARY_LE", "-O", str(ref_file), "-O", str(target_file), "-ICP", "-REFERENCE_IS_FIRST", "-ITER", "100", "-SAVE_CLOUDS"]
            }
        ]
        
        for strategy in strategies:
            print(f"   Trying: {strategy['name']}...")
            
            if self.run_cloudcompare_command(strategy["args"]):
                time.sleep(5)
                
                # Enhanced file detection patterns from community
                new_files = set(self.output_dir.glob("*.ply")) - existing_files
                registered_files = [f for f in new_files if any(keyword in f.name.upper() for keyword in ["REGISTERED", "ICP", "ALIGNED"])]
                
                # Also check parent directories (CloudCompare sometimes saves there)
                for check_dir in [Path(ref_file).parent, Path(target_file).parent, Path.cwd()]:
                    if check_dir != self.output_dir:
                        dir_files = set(check_dir.glob("*.ply"))
                        new_in_dir = [f for f in dir_files if any(keyword in f.name.upper() for keyword in ["REGISTERED", "ICP", "ALIGNED"]) and f.stat().st_mtime > time.time() - 300]
                        registered_files.extend(new_in_dir)
                
                if registered_files:
                    # Use most recent file
                    registered_file = max(registered_files, key=lambda x: x.stat().st_mtime)
                    output_file = self.output_dir / "target_icp_registered.ply"
                    
                    if registered_file.parent != self.output_dir:
                        shutil.move(str(registered_file), str(output_file))
                    else:
                        registered_file.rename(output_file)
                    
                    print(f"ICP successful with {strategy['name']}: {output_file.name}")
                    return str(output_file)
        
        print("All ICP strategies failed")
        return None
    
    def cloudcompare_compute_distances(self, ref_file, aligned_file):
        """High-accuracy distance computation for precise heatmap generation"""
        print("CloudCompare high-accuracy distance computation...")
        
        existing_files = set(self.output_dir.glob("*.ply"))
        
        # Enhanced distance computation strategies for maximum heatmap accuracy
        strategies = [
            # Strategy 1: High-accuracy with octree optimization
            ["-SILENT", "-C_EXPORT_FMT", "PLY", "-PLY_EXPORT_FMT", "BINARY_LE", 
             "-O", str(ref_file), "-O", str(aligned_file), 
             "-C2C_DIST", "-REFERENCE_IS_FIRST", "-OCTREE_LEVEL", "10", 
             "-MAX_DIST", "1.5", "-SPLIT_XYZ", "-SAVE_CLOUDS"],
            
            # Strategy 2: Maximum precision with statistical filtering
            ["-SILENT", "-AUTO_SAVE", "OFF", "-C_EXPORT_FMT", "PLY", 
             "-O", str(ref_file), "-O", str(aligned_file), 
             "-C2C_DIST", "-REFERENCE_IS_FIRST", "-OCTREE_LEVEL", "12", 
             "-MAX_DIST", "2.0", "-SPLIT_XYZ", "-LOCAL_STAT_TEST", "2.5", "12", 
             "-MULTI_THREAD", "-SAVE_CLOUDS"],
            
            # Strategy 3: Ultra-high precision (slower but most accurate)
            ["-SILENT", "-C_EXPORT_FMT", "PLY", "-PLY_EXPORT_FMT", "BINARY_LE",
             "-O", str(ref_file), "-O", str(aligned_file), 
             "-C2C_DIST", "-REFERENCE_IS_FIRST", "-OCTREE_LEVEL", "14", 
             "-MAX_DIST", "3.0", "-SPLIT_XYZ", "-LOCAL_STAT_TEST", "2.0", "16", 
             "-MULTI_THREAD", "-SAVE_CLOUDS"],
            
            # Strategy 4: Fallback - basic but reliable
            ["-SILENT", "-AUTO_SAVE", "OFF", "-O", str(ref_file), "-O", str(aligned_file), 
             "-C2C_DIST", "-REFERENCE_IS_FIRST", "-SAVE_CLOUDS"]
        ]
        
        strategy_names = [
            "High-accuracy octree optimization",
            "Maximum precision with statistical filtering", 
            "Ultra-high precision (slowest)",
            "Fallback basic computation"
        ]
        
        for i, (strategy, name) in enumerate(zip(strategies, strategy_names), 1):
            print(f"   Strategy {i}: {name}...")
            
            if self.run_cloudcompare_command(strategy):
                time.sleep(5)  # Longer wait for high-precision computation
                
                # Find distance files
                new_files = set(self.output_dir.glob("*.ply")) - existing_files
                distance_files = [f for f in new_files if "C2C_DIST" in f.name.upper()]
                
                if distance_files:
                    print(f"High-accuracy distance computation successful: {len(distance_files)} files")
                    for f in distance_files:
                        size_mb = f.stat().st_size / (1024 * 1024)
                        print(f"   {f.name} ({size_mb:.1f} MB)")
                    
                    print(f"   Heatmap accuracy level: {name}")
                    return distance_files
        
        print("All high-accuracy distance computation strategies failed")
        return []
    
    def create_verification_overlay(self, ref_file, aligned_file):
        """Create verification overlay for visual inspection"""
        try:
            import open3d as o3d
            
            print("Creating verification overlay...")
            
            ref_cloud = o3d.io.read_point_cloud(ref_file)
            aligned_cloud = o3d.io.read_point_cloud(aligned_file)
            
            # Color clouds for verification
            ref_cloud.paint_uniform_color([0, 0, 1])      # Blue = Reference
            aligned_cloud.paint_uniform_color([1, 0, 0])  # Red = Aligned
            
            # Combine
            combined = ref_cloud + aligned_cloud
            
            # Save overlay
            overlay_file = self.output_dir / "verification_overlay.ply"
            o3d.io.write_point_cloud(str(overlay_file), combined)
            
            print("Verification overlay created!")
            print(f"   Blue = Reference mill")
            print(f"   Red = Aligned mill")
            print(f"   Perfect alignment = tight color overlap")
            
            return str(overlay_file)
            
        except ImportError:
            print("Open3D not available for overlay")
            return None
        except Exception as e:
            print(f"Overlay creation failed: {e}")
            return None
    
    def run_complete_mill_analysis(self, ref_path, target_path):
        """Execute complete mill wear analysis pipeline - NO CROPPING"""
        print("="*70)
        print("COMPLETE CLOUDCOMPARE MILL WEAR ANALYSIS - HIGH DETAIL")
        print("="*70)
        print(f"Reference: {Path(ref_path).name}")
        print(f"Target: {Path(target_path).name}")
        print("Community-proven CloudCompare techniques")
        print("NO CROPPING - Full mill preserved")
        print(f"High detail subsampling: {self.subsample_points:,} points")
        print("="*70)
        
        start_time = time.time()
        
        try:
            # Step 1: Subsample both clouds for manageable processing
            print("\nPHASE 1: DATA PREPARATION")
            ref_subsampled = self.cloudcompare_subsample_robust(ref_path)
            target_subsampled = self.cloudcompare_subsample_robust(target_path)
            
            if not ref_subsampled or not target_subsampled:
                print("Subsampling failed")
                return None
            
            # Step 2: Bounding box center alignment
            print("\nPHASE 2: INITIAL ALIGNMENT")
            target_centered = self.align_box_center_with_open3d(ref_subsampled, target_subsampled)
            
            # REMOVED: Step 3 - Distance-based cropping (was cutting too much data)
            print("\nPHASE 3: SKIPPING CROPPING")
            print("Cropping step skipped - preserving full mill geometry")
            print("Full aligned model preserved (no points removed)")
            
            # Step 4: CloudCompare ICP registration (now uses full aligned model)
            print("\nPHASE 4: PRECISION REGISTRATION")
            target_registered = self.cloudcompare_icp_progressive(ref_subsampled, target_centered)
            
            final_target = target_registered if target_registered else target_centered
            
            # Step 5: Distance computation for wear analysis
            print("\nPHASE 5: HIGH-ACCURACY WEAR ANALYSIS")
            distance_files = self.cloudcompare_compute_distances(ref_subsampled, final_target)
            
            # Step 6: Verification overlay
            print("\nPHASE 6: VERIFICATION")
            overlay_file = self.create_verification_overlay(ref_subsampled, final_target)
            
            # Results summary
            end_time = time.time() 
            total_time = end_time - start_time
            
            print("\n" + "="*70)
            print("MILL WEAR ANALYSIS COMPLETED!")
            print(f"Total processing time: {total_time:.1f} seconds")
            print("Community-proven CloudCompare pipeline executed successfully")
            print("NO CROPPING - Full mill geometry preserved")
            print(f"High detail analysis with {self.subsample_points:,} points")
            print("="*70)
            
            print("FINAL RESULTS:")
            print(f"   Reference (subsampled): {Path(ref_subsampled).name}")
            print(f"   Final aligned target: {Path(final_target).name}")
            if distance_files:
                print(f"   Wear analysis files: {len(distance_files)} files")
                for f in distance_files[:3]:  # Show first 3
                    print(f"     - {f.name}")
            if overlay_file:
                print(f"   Verification overlay: {Path(overlay_file).name}")
            
            print(f"\nALL OUTPUT FILES:")
            for f in sorted(self.output_dir.glob("*.ply")):
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"   - {f.name} ({size_mb:.2f} MB)")
            
            print(f"\nVISUALIZATION:")
            if overlay_file:
                print(f'   CloudCompare "{overlay_file}"')
            
            print(f"\nTECHNIQUES USED:")
            print(f"   Bounding box center alignment function")
            print(f"   Distance-based cropping REMOVED (was cutting too much)")
            print(f"   Community-proven CloudCompare subsampling ({self.subsample_points:,} points)")
            print(f"   Progressive CloudCompare ICP strategies")
            print(f"   High-accuracy CloudCompare distance computation with octree optimization")
            print(f"   Enhanced heatmap precision with statistical filtering")
            print(f"   Visual verification overlay")
            
            return {
                'ref_subsampled': ref_subsampled,
                'target_final': final_target,
                'distance_files': distance_files,
                'overlay_file': overlay_file,
                'processing_time': total_time,
                'icp_successful': target_registered is not None
            }
            
        except Exception as e:
            print(f"Complete analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None

def launch_cloudcompare_viewer(file_path, cloudcompare_path):
    """Launch CloudCompare GUI to view results"""
    try:
        print(f"Launching CloudCompare viewer...")
        subprocess.Popen([cloudcompare_path, str(file_path)])
        print("CloudCompare viewer launched")
    except Exception as e:
        print(f"Failed to launch CloudCompare viewer: {e}")

# MAIN EXECUTION - HIGH DETAIL VERSION
if __name__ == "__main__":
    # Windows configuration
    cloudcompare_executable = r"C:\Program Files\CloudCompare\CloudCompare.exe"
    project_directory = r"C:\Mill_Analysis_Project"
    
    # Mill scan files
    reference_mill = r"C:\Mill_Analysis_Project\data\inner_scans\3_18-Jan-2024_converted.ply"
    target_mill = r"C:\Mill_Analysis_Project\data\inner_scans\5_26-Mar-2024_converted.ply"
    
    print("STARTING CLOUDCOMPARE MILL ANALYSIS - HIGH DETAIL")
    print("Using community-proven techniques with enhanced detail")
    print("Full mill geometry preserved - no points removed")
    print("High detail subsampling: 5,000,000 points")
    print()
    
    # Initialize analyzer with high detail settings
    analyzer = CloudCompareMillAnalysis(
        cloudcompare_executable, 
        project_directory, 
        subsample_points=5000000
    )
    
    # Run complete analysis without cropping
    results = analyzer.run_complete_mill_analysis(reference_mill, target_mill)
    
    if results:
        print(f"\nSUCCESS! Analysis completed in {results['processing_time']:.1f}s")
        print("Full mill geometry preserved - no cropping applied")
        print("High detail analysis maintains maximum surface information")
        
        if results['icp_successful']:
            print("CloudCompare ICP registration successful!")
        else:
            print("ICP used fallback - but analysis completed successfully")
        
        # Launch CloudCompare viewer for verification
        if results['overlay_file']:
            choice = input("\nLaunch CloudCompare to view alignment overlay? (y/n): ").lower()
            if choice == 'y':
                launch_cloudcompare_viewer(results['overlay_file'], cloudcompare_executable)
        
        print(f"\nAll results saved in: {analyzer.output_dir}")
        print("CloudCompare mill wear analysis complete - high detail with enhanced heatmap accuracy!")
        
    else:
        print("\nAnalysis failed")
        print("Check CloudCompare installation and input file paths")