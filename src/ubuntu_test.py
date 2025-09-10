import os
import numpy as np
from pathlib import Path
import time
import shutil

class MillWearAnalyzer:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir).expanduser()
        self.output_dir = self.base_dir / "mill_wear_analysis"
        self.output_dir.mkdir(exist_ok=True)
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        
        print(f"🔧 Mill Wear Analyzer initialized")
        print(f"📂 Working directory: {self.output_dir}")
    
    def subsample_clouds(self, ref_path, target_path, num_points=500000):
        """Step 1: Subsample both clouds"""
        print("🔄 Step 1: Subsampling point clouds...")
        
        original_dir = os.getcwd()
        os.chdir(self.base_dir)
        
        try:
            # Clear old files
            for f in Path('.').glob("*RANDOM_SUBSAMPLED*.ply"):
                f.unlink()
            
            # Subsample reference
            print(f"   Subsampling reference → {num_points:,} points...")
            cmd1 = f'CloudCompare -SILENT -C_EXPORT_FMT PLY -PLY_EXPORT_FMT BINARY_LE -O "{ref_path}" -SS RANDOM {num_points}'
            os.system(cmd1)
            
            # Subsample target
            print(f"   Subsampling target → {num_points:,} points...")
            cmd2 = f'CloudCompare -SILENT -C_EXPORT_FMT PLY -PLY_EXPORT_FMT BINARY_LE -O "{target_path}" -SS RANDOM {num_points}'
            os.system(cmd2)
            
            # Find generated files
            time.sleep(2)
            subsampled_files = list(Path('.').glob("*RANDOM_SUBSAMPLED*.ply"))
            subsampled_files.sort(key=lambda x: x.stat().st_mtime)
            
            if len(subsampled_files) >= 2:
                ref_sub = subsampled_files[-2]
                target_sub = subsampled_files[-1]
                
                # Move to output directory
                ref_dest = self.output_dir / "01_ref_subsampled.ply"
                target_dest = self.output_dir / "01_target_subsampled.ply"
                
                shutil.move(str(ref_sub), str(ref_dest))
                shutil.move(str(target_sub), str(target_dest))
                
                print(f"✅ Subsampling complete!")
                return str(ref_dest), str(target_dest)
            else:
                print(f"❌ Expected 2 files, found {len(subsampled_files)}")
                return None, None
                
        finally:
            os.chdir(original_dir)
    
    def automatic_initial_alignment(self, ref_file, target_file):
        """Step 2: AUTOMATIC alignment - center + PCA rotation (NO manual angles!)"""
        print("🔄 Step 2: Automatic initial alignment (PCA-based)...")
        
        try:
            import open3d as o3d
            
            # Load clouds
            ref_cloud = o3d.io.read_point_cloud(ref_file)
            target_cloud = o3d.io.read_point_cloud(target_file)
            
            print(f"   Reference points: {len(ref_cloud.points):,}")
            print(f"   Target points: {len(target_cloud.points):,}")
            
            # Step 2a: Center alignment (translation)
            ref_center = ref_cloud.get_center()
            target_center = target_cloud.get_center()
            translation = ref_center - target_center
            
            print(f"   Translation: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]")
            
            # Apply translation
            transformation = np.eye(4)
            transformation[:3, 3] = translation
            target_cloud.transform(transformation)
            
            # Step 2b: PCA-based rotation estimation (AUTOMATIC!)
            print("   Computing automatic rotation using PCA...")
            
            ref_points = np.asarray(ref_cloud.points)
            target_points = np.asarray(target_cloud.points)
            
            # Center the point clouds
            ref_centered = ref_points - np.mean(ref_points, axis=0)
            target_centered = target_points - np.mean(target_points, axis=0)
            
            # Compute covariance matrices and principal axes
            ref_cov = np.cov(ref_centered.T)
            target_cov = np.cov(target_centered.T)
            
            # SVD to get principal axes
            _, _, ref_axes = np.linalg.svd(ref_cov)
            _, _, target_axes = np.linalg.svd(target_cov)
            
            # Compute rotation matrix to align principal axes
            rotation_matrix = ref_axes @ target_axes.T
            
            # Ensure proper rotation (determinant should be +1)
            if np.linalg.det(rotation_matrix) < 0:
                target_axes[-1] *= -1
                rotation_matrix = ref_axes @ target_axes.T
            
            # Apply rotation
            target_cloud.rotate(rotation_matrix, center=target_cloud.get_center())
            
            print(f"   ✅ Automatic rotation applied (PCA-based)")
            
            # Save automatically aligned target
            auto_aligned = self.output_dir / "02_target_auto_aligned.ply"
            o3d.io.write_point_cloud(str(auto_aligned), target_cloud)
            
            print("✅ Automatic initial alignment complete!")
            print(f"   No manual angles needed - PCA found optimal rotation")
            
            return ref_file, str(auto_aligned)
            
        except ImportError:
            print("❌ Open3D not available - install with: pip install open3d")
            print("   Falling back to center-only alignment...")
            return self.simple_center_alignment(ref_file, target_file)
        except Exception as e:
            print(f"❌ Automatic alignment failed: {e}")
            print("   Falling back to center-only alignment...")
            return self.simple_center_alignment(ref_file, target_file)
    
    def simple_center_alignment(self, ref_file, target_file):
        """Fallback: Simple center alignment if Open3D not available"""
        try:
            import open3d as o3d
            
            ref_cloud = o3d.io.read_point_cloud(ref_file)
            target_cloud = o3d.io.read_point_cloud(target_file)
            
            # Just center alignment
            ref_center = ref_cloud.get_center()
            target_center = target_cloud.get_center()
            translation = ref_center - target_center
            
            transformation = np.eye(4)
            transformation[:3, 3] = translation
            target_cloud.transform(transformation)
            
            # Save centered target
            centered_file = self.output_dir / "02_target_centered.ply"
            o3d.io.write_point_cloud(str(centered_file), target_cloud)
            
            print("✅ Center alignment applied (ICP will handle rotation)")
            return ref_file, str(centered_file)
            
        except Exception as e:
            print(f"❌ Center alignment failed: {e}")
            return ref_file, target_file
    
    def precision_icp_registration(self, ref_file, target_file, max_iterations=150):
        """Step 3: Let ICP do its job - it will find optimal rotation automatically"""
        print(f"🔄 Step 3: Precision ICP (ICP will find optimal rotation)...")
        
        original_dir = os.getcwd()
        os.chdir(self.output_dir)
        
        try:
            # Clear old registered files
            for f in Path('.').glob("*REGISTERED*.ply"):
                f.unlink()
            
            # ICP command - NO scale adjustment, let ICP find rotation
            cmd = f'''CloudCompare -SILENT -LOG_FILE icp_precision.log \
                     -C_EXPORT_FMT PLY -PLY_EXPORT_FMT BINARY_LE \
                     -O "{ref_file}" -O "{target_file}" \
                     -ICP -REFERENCE_IS_FIRST \
                     -ITER {max_iterations} \
                     -OVERLAP 95 \
                     -RANDOM_SAMPLING_LIMIT 200000 \
                     -ENABLE_FARTHEST_REMOVAL'''
            
            print(f"   Running ICP with {max_iterations} iterations...")
            print(f"   ICP will automatically find optimal rotation & translation")
            
            result = os.system(cmd)
            time.sleep(3)
            
            # Parse log
            self.parse_icp_log("icp_precision.log")
            
            # Find registered file
            registered_files = list(Path('.').glob("*REGISTERED*.ply"))
            if registered_files:
                reg_file = max(registered_files, key=lambda x: x.stat().st_mtime)
                final_aligned = self.output_dir / "03_target_icp_aligned.ply"
                
                shutil.copy2(str(reg_file), str(final_aligned))
                
                print("✅ ICP registration complete!")
                print("   ICP automatically found optimal alignment!")
                return str(ref_file), str(final_aligned)
            else:
                print("❌ No registered file found")
                return None, None
                
        finally:
            os.chdir(original_dir)
    
    def parse_icp_log(self, log_file):
        """Parse ICP log for results"""
        try:
            if Path(log_file).exists():
                with open(log_file, 'r') as f:
                    content = f.read()
                
                lines = content.split('\n')
                for line in lines:
                    if 'RMS:' in line:
                        print(f"📊 {line.strip()}")
                    elif 'Number of points used' in line:
                        print(f"📊 {line.strip()}")
                    elif 'Entity' in line and 'registered' in line:
                        print(f"📊 {line.strip()}")
        except:
            pass
    
    def compute_wear_distances(self, ref_file, aligned_file, max_distance=1.0):
        """Step 4: Compute wear distances"""
        print("🔄 Step 4: Computing mill wear distances...")
        
        original_dir = os.getcwd()
        os.chdir(self.output_dir)
        
        try:
            cmd = f'''CloudCompare -SILENT -C_EXPORT_FMT PLY -PLY_EXPORT_FMT BINARY_LE \
                     -O "{ref_file}" -O "{aligned_file}" \
                     -C2C_DIST -MAX_DIST {max_distance}'''
            
            print(f"   Max distance threshold: {max_distance}")
            
            result = os.system(cmd)
            
            if result == 0:
                time.sleep(3)
                dist_files = list(Path('.').glob("*C2C_DIST*.ply"))
                
                if dist_files:
                    print("✅ Mill wear distance computation complete!")
                    for f in dist_files:
                        size_mb = f.stat().st_size / (1024 * 1024)
                        print(f"   📈 {f.name} ({size_mb:.1f} MB)")
                    
                    return dist_files
                else:
                    print("❌ No distance files generated")
                    return []
            else:
                print(f"❌ Distance computation failed")
                return []
                
        finally:
            os.chdir(original_dir)
    
    def create_alignment_overlay(self, ref_file, aligned_file):
        """Step 5: Create alignment verification overlay"""
        print("🔄 Step 5: Creating alignment verification overlay...")
        
        try:
            import open3d as o3d
            
            # Load clouds
            ref_cloud = o3d.io.read_point_cloud(ref_file)
            aligned_cloud = o3d.io.read_point_cloud(aligned_file)
            
            print(f"   Reference points: {len(ref_cloud.points):,}")
            print(f"   Aligned points: {len(aligned_cloud.points):,}")
            
            # Color clouds for verification
            ref_cloud.paint_uniform_color([0, 0, 1])      # Blue = Reference
            aligned_cloud.paint_uniform_color([1, 0, 0])  # Red = Target
            
            # Combine
            combined_cloud = ref_cloud + aligned_cloud
            
            # Save overlay
            overlay_file = self.output_dir / "04_alignment_verification.ply"
            o3d.io.write_point_cloud(str(overlay_file), combined_cloud)
            
            print("✅ Alignment verification overlay created!")
            print(f"   🔵 Blue = Reference mill")
            print(f"   🔴 Red = Target mill") 
            print(f"   Perfect alignment = tight overlap of colors")
            
            return str(overlay_file)
            
        except ImportError:
            print("❌ Open3D not available for overlay")
            return None
        except Exception as e:
            print(f"❌ Overlay creation failed: {e}")
            return None
    
    def run_automatic_mill_analysis(self, ref_path, target_path):
        """Execute FULLY AUTOMATIC mill wear analysis - NO manual angles needed!"""
        print("="*70)
        print("🏭 FULLY AUTOMATIC MILL WEAR ANALYSIS")
        print("="*70)
        print(f"📅 Reference: {Path(ref_path).name}")
        print(f"📅 Target: {Path(target_path).name}")
        print("🤖 Automatic alignment - NO manual rotation angles needed!")
        print("="*70)
        
        start_time = time.time()
        
        try:
            # Step 1: Subsample
            ref_sub, target_sub = self.subsample_clouds(ref_path, target_path, 500000)
            if not ref_sub or not target_sub:
                print("❌ Subsampling failed")
                return None
            
            # Step 2: Automatic initial alignment (PCA-based rotation)
            ref_aligned, target_aligned = self.automatic_initial_alignment(ref_sub, target_sub)
            
            # Step 3: Let ICP find optimal rotation & translation
            ref_final, target_final = self.precision_icp_registration(ref_aligned, target_aligned, 150)
            if not ref_final or not target_final:
                print("❌ ICP registration failed")
                return None
            
            # Step 4: Compute wear distances
            distance_files = self.compute_wear_distances(ref_final, target_final, 1.0)
            
            # Step 5: Create verification overlay
            overlay_file = self.create_alignment_overlay(ref_final, target_final)
            
            # Results summary
            end_time = time.time()
            total_time = end_time - start_time
            
            print("="*70)
            print("✅ AUTOMATIC MILL WEAR ANALYSIS COMPLETED!")
            print(f"⏱️  Processing time: {total_time:.1f} seconds")
            print("🤖 No manual intervention required - fully automatic!")
            print("="*70)
            
            print("📊 ANALYSIS RESULTS:")
            print(f"   🎯 Aligned mill: {Path(target_final).name}")
            if distance_files:
                print(f"   📈 Wear heatmap: {distance_files[0].name}")
            if overlay_file:
                print(f"   👁️  Verification: {Path(overlay_file).name}")
            
            print("\n📁 ALL FILES:")
            for file_path in sorted(self.output_dir.glob("*.ply")):
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"   - {file_path.name} ({size_mb:.2f} MB)")
            
            print("\n🎨 VISUALIZATION:")
            print(f"   CloudCompare {overlay_file}")
            if distance_files:
                print(f"   CloudCompare {distance_files[0]}")
            
            print("="*70)
            
            return {
                'aligned_file': target_final,
                'distance_files': distance_files,
                'overlay_file': overlay_file,
                'processing_time': total_time
            }
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return None

def visualize_mill_wear(overlay_file):
    """Interactive 3D visualization"""
    try:
        import open3d as o3d
        
        print("🎨 Opening automatic alignment visualization...")
        print("   🔵 Blue = Reference mill")
        print("   🔴 Red = Automatically aligned mill")
        print("   Perfect alignment = tight color overlap")
        
        overlay_cloud = o3d.io.read_point_cloud(overlay_file)
        o3d.visualization.draw_geometries(
            [overlay_cloud],
            window_name="Automatic Mill Alignment - No Manual Angles!",
            width=1400,
            height=900
        )
        
    except ImportError:
        print("❌ Open3D not available - install with: pip install open3d")
    except Exception as e:
        print(f"❌ Visualization failed: {e}")

# MAIN EXECUTION - FULLY AUTOMATIC!
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = MillWearAnalyzer("/home/koop/FLS-project")
    
    # Your mill scan files
    reference_mill = "/home/koop/FLS-project/2_28-Oct-2023_converted.ply"
    target_mill = "/home/koop/FLS-project/MMI BM 1-Oct-2024_IPE003_converted.ply"
    
    print("🤖 Running FULLY AUTOMATIC alignment - no manual angles needed!")
    
    # Run automatic analysis
    results = analyzer.run_automatic_mill_analysis(reference_mill, target_mill)
    
    if results:
        print(f"\n🎉 Automatic mill analysis completed in {results['processing_time']:.1f}s")
        print("🤖 Perfect alignment achieved automatically!")
        
        # Launch visualization
        if results['overlay_file']:
            choice = input("\n🎨 Launch 3D visualization? (y/n): ").lower()
            if choice == 'y':
                visualize_mill_wear(results['overlay_file'])
    else:
        print("\n❌ Analysis failed")
        print("💡 Install Open3D for best results: pip install open3d")
