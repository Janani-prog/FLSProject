import open3d as o3d
import numpy as np
import copy
from typing import Optional, Tuple, Dict, Any, List
from sklearn.decomposition import PCA

from config import ProcessingConfig

#edited
def manual_gap_xyz_alignment(reference_pcd, scan_pcd):
    import numpy as np
    import copy

    ref_pts = np.asarray(reference_pcd.points)
    scan_pts = np.asarray(scan_pcd.points)

    # X-axis center (equal gaps left/right)
    x_center_ref = (ref_pts[:, 0].min() + ref_pts[:, 0].max()) / 2
    x_center_scan = (scan_pts[:, 0].min() + scan_pts[:, 0].max()) / 2
    x_shift = x_center_ref - x_center_scan

    # Z-axis top surface alignment (level at top)
    ref_top_z = np.mean(ref_pts[ref_pts[:, 2] >= np.percentile(ref_pts[:, 2], 95)][:, 2])
    scan_top_z = np.mean(scan_pts[scan_pts[:, 2] >= np.percentile(scan_pts[:, 2], 95)][:, 2])
    z_shift = ref_top_z - scan_top_z

    # Apply translation
    new_pcd = copy.deepcopy(scan_pcd)
    new_pcd = new_pcd.translate([x_shift, 0, z_shift])

    return new_pcd


class AlignmentProcessor:
    """Complete enhanced alignment processor with all features built-in."""
    
    def __init__(self, enable_fixed_scaling: bool = True, fixed_scale_factor: float = 1000.0):
        """
        Initialize alignment processor with enhanced capabilities.
        
        Args:
            enable_fixed_scaling: Whether to use fixed scaling instead of automatic scaling
            fixed_scale_factor: Fixed scale factor to apply (default: 1000.0)
        """
        self.config = ProcessingConfig()
        self.enable_fixed_scaling = enable_fixed_scaling
        self.fixed_scale_factor = fixed_scale_factor
        
        # Track scaling results for consistency
        self.scaling_results = {}
        
        # Store dual-scan data for post-processing ICP
        self.dual_scan_data = {}
        self.post_icp_results = None
    
    def apply_fixed_scaling(self, pcd: o3d.geometry.PointCloud, 
                           scan_identifier: str = "unknown") -> Tuple[o3d.geometry.PointCloud, float]:
        """Apply consistent fixed scaling instead of automatic 95th percentile scaling."""
        if not self.enable_fixed_scaling:
            return copy.deepcopy(pcd), 1.0
            
        print(f"Applying FIXED scaling (factor: {self.fixed_scale_factor}) to {scan_identifier}")
        
        scaled_pcd = copy.deepcopy(pcd)
        scaled_pcd.scale(self.fixed_scale_factor, center=scaled_pcd.get_center())
        
        original_bbox = pcd.get_axis_aligned_bounding_box()
        scaled_bbox = scaled_pcd.get_axis_aligned_bounding_box()
        original_dims = original_bbox.get_extent()
        scaled_dims = scaled_bbox.get_extent()
        
        print(f"Fixed scaling applied to {scan_identifier}:")
        print(f"  Original dimensions: {original_dims[0]:.1f} x {original_dims[1]:.1f} x {original_dims[2]:.1f}")
        print(f"  Scaled dimensions: {scaled_dims[0]:.1f} x {scaled_dims[1]:.1f} x {scaled_dims[2]:.1f}")
        print(f"  Scale factor applied: {self.fixed_scale_factor}")
        
        self.scaling_results[scan_identifier] = {
            'scale_factor': self.fixed_scale_factor,
            'original_dims': original_dims,
            'scaled_dims': scaled_dims
        }
        
        return scaled_pcd, self.fixed_scale_factor
    
    def apply_center_alignment(self, reference_pcd: o3d.geometry.PointCloud,
                             scaled_inner_pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Apply center alignment using old_version algorithm."""
        print(f"\nApplying center alignment (old_version algorithm)...")
        
        ref_points = np.asarray(reference_pcd.points)
        scaled_points = np.asarray(scaled_inner_pcd.points)
        
        ref_center = np.mean(ref_points, axis=0)
        scaled_center = np.mean(scaled_points, axis=0)
        
        print(f"Reference center: [{ref_center[0]:.2f}, {ref_center[1]:.2f}, {ref_center[2]:.2f}]")
        print(f"Scaled inner center: [{scaled_center[0]:.2f}, {scaled_center[1]:.2f}, {scaled_center[2]:.2f}]")
        
        translation = ref_center - scaled_center
        translation_distance = np.linalg.norm(translation)
        
        print(f"Translation needed: [{translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f}]")
        print(f"Translation distance: {translation_distance:.2f}mm")
        
        center_aligned_pcd = copy.deepcopy(scaled_inner_pcd)
        
        if translation_distance > 0.1:
            center_aligned_pcd = center_aligned_pcd.translate(translation)
            print("SUCCESS: Center alignment applied")
        else:
            print("INFO: No significant translation needed")
        
        aligned_center = center_aligned_pcd.get_center()
        print(f"Final center: [{aligned_center[0]:.2f}, {aligned_center[1]:.2f}, {aligned_center[2]:.2f}]")
        
        verification_distance = np.linalg.norm(aligned_center - ref_center)
        print(f"Center alignment accuracy: {verification_distance:.3f}mm")
        
        return center_aligned_pcd
    
    def apply_z_axis_alignment(self, reference_pcd: o3d.geometry.PointCloud,
                              center_aligned_pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Apply Z-axis alignment for mill surface positioning."""
        print(f"\nApplying Z-axis surface alignment (old_version algorithm)...")
        
        try:
            ref_points = np.asarray(reference_pcd.points)
            aligned_points = np.asarray(center_aligned_pcd.points)
            
            ref_heights = ref_points[:, 2]
            ref_top_threshold = np.percentile(ref_heights, 95)
            ref_top_mask = ref_heights >= ref_top_threshold
            ref_top_points = ref_points[ref_top_mask]
            
            inner_heights = aligned_points[:, 2]
            inner_surface_threshold = np.percentile(inner_heights, 95)
            inner_surface_mask = inner_heights >= inner_surface_threshold
            inner_surface_points = aligned_points[inner_surface_mask]
            
            print(f"Reference top surface points: {len(ref_top_points):,}")
            print(f"Inner surface points: {len(inner_surface_points):,}")
            
            if len(ref_top_points) > 100 and len(inner_surface_points) > 100:
                ref_top_z = np.mean(ref_top_points[:, 2])
                inner_top_z = np.mean(inner_surface_points[:, 2])
                
                print(f"Reference top surface Z: {ref_top_z:.1f}mm")
                print(f"Inner top surface Z: {inner_top_z:.1f}mm")
                
                #edited
                lower_by = 21.75
                height_adjustment = ref_top_z - inner_top_z - lower_by
                translation = np.array([0, 0, height_adjustment])
                
                z_aligned_pcd = copy.deepcopy(center_aligned_pcd)
                z_aligned_pcd = z_aligned_pcd.translate(translation)
                
                final_points = np.asarray(z_aligned_pcd.points)
                final_heights = final_points[:, 2]
                final_top_z = np.percentile(final_heights, 95)
                final_top_mean = np.mean(final_points[final_heights >= final_top_z][:, 2])
                surface_separation = ref_top_z - final_top_mean
                
                print(f"Surface-based positioning applied:")
                print(f"  Height adjustment: {height_adjustment:.2f}mm")
                print(f"  Reference top surface Z: {ref_top_z:.1f}mm")
                print(f"  Inner surface positioned at Z: {final_top_mean:.1f}mm")
                print(f"  Surface separation: {surface_separation:.1f}mm")
                print("SUCCESS: Z-axis surface alignment applied")
                
                return z_aligned_pcd
            else:
                print("Insufficient surface points for positioning, applying fallback...")
                return self._apply_fallback_z_alignment(reference_pcd, center_aligned_pcd)
                
        except Exception as e:
            print(f"Surface positioning failed: {e}")
            return self._apply_fallback_z_alignment(reference_pcd, center_aligned_pcd)
    
    def _apply_fallback_z_alignment(self, reference_pcd: o3d.geometry.PointCloud,
                                   center_aligned_pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Fallback Z-axis alignment method."""
        print("Applying fallback Z-axis positioning...")
        
        try:
            ref_points = np.asarray(reference_pcd.points)
            ref_center = reference_pcd.get_center()
            scaled_center = center_aligned_pcd.get_center()
            
            ref_top_z = np.percentile(ref_points[:, 2], 95)
            
            translation = ref_center - scaled_center
            translation[2] = ref_top_z - scaled_center[2] - 100
            
            z_aligned_pcd = copy.deepcopy(center_aligned_pcd)
            z_aligned_pcd = z_aligned_pcd.translate(translation)
            
            print(f"Fallback positioning applied:")
            print(f"  Reference top Z: {ref_top_z:.1f}mm")
            print(f"  Positioned 100mm below reference top surface")
            print("Applied fallback positioning with top surface alignment")
            
            return z_aligned_pcd
            
        except Exception as e:
            print(f"Fallback positioning failed: {e}, using center alignment")
            return center_aligned_pcd
    
    def apply_pca_axis_alignment(self, reference_pcd: o3d.geometry.PointCloud,
                           z_axis_aligned_pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Apply PCA-based axis alignment."""
        print(f"\nApplying PCA axis alignment (old_version algorithm)...")
        
        ref_points = np.asarray(reference_pcd.points)
        aligned_points = np.asarray(z_axis_aligned_pcd.points)
        
        ref_center = np.mean(ref_points, axis=0)
        aligned_center = np.mean(aligned_points, axis=0)
        
        ref_pca = PCA(n_components=3)
        aligned_pca = PCA(n_components=3)
        
        ref_pca.fit(ref_points - ref_center)
        aligned_pca.fit(aligned_points - aligned_center)
        
        ref_axis = ref_pca.components_[0]
        aligned_axis = aligned_pca.components_[0]
        
        axis_alignment = np.abs(np.dot(ref_axis, aligned_axis))
        print(f"Current axis alignment: {axis_alignment:.3f}")
        
        print("Applying axis rotation...")
        
        try:
            ref_axis_norm = ref_axis / np.linalg.norm(ref_axis)
            aligned_axis_norm = aligned_axis / np.linalg.norm(aligned_axis)
            
            rotation_axis = np.cross(aligned_axis_norm, ref_axis_norm)
            rotation_axis_norm = np.linalg.norm(rotation_axis)
            
            if rotation_axis_norm > 1e-6:
                rotation_axis = rotation_axis / rotation_axis_norm
                
                dot_product = np.dot(aligned_axis_norm, ref_axis_norm)
                angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
                
                print(f"Rotation angle: {np.degrees(angle):.2f} degrees")
                
                cos_angle = np.cos(angle)
                sin_angle = np.sin(angle)
                
                K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                            [rotation_axis[2], 0, -rotation_axis[0]],
                            [-rotation_axis[1], rotation_axis[0], 0]])
                
                rotation_matrix = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)
                
                axis_aligned_pcd = copy.deepcopy(z_axis_aligned_pcd)
                axis_aligned_pcd = axis_aligned_pcd.translate(-aligned_center)
                
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = rotation_matrix
                axis_aligned_pcd = axis_aligned_pcd.transform(transform_matrix)
                
                axis_aligned_pcd = axis_aligned_pcd.translate(aligned_center)
                
                final_points = np.asarray(axis_aligned_pcd.points)
                final_center = np.mean(final_points, axis=0)
                final_pca = PCA(n_components=3)
                final_pca.fit(final_points - final_center)
                final_axis = final_pca.components_[0]
                
                final_alignment = np.abs(np.dot(ref_axis, final_axis))
                print(f"Final axis alignment: {final_alignment:.3f}")
                print("SUCCESS: Axis alignment applied")
                
                return axis_aligned_pcd
                
            else:
                dot_product = np.dot(aligned_axis_norm, ref_axis_norm)
                if dot_product < 0:
                    print("Axes are anti-parallel, applying 180-degree rotation")
                    
                    if abs(ref_axis_norm[0]) < 0.9:
                        perp_axis = np.array([1, 0, 0])
                    else:
                        perp_axis = np.array([0, 1, 0])
                    
                    perp_axis = perp_axis - np.dot(perp_axis, ref_axis_norm) * ref_axis_norm
                    perp_axis = perp_axis / np.linalg.norm(perp_axis)
                    
                    rotation_matrix = 2 * np.outer(perp_axis, perp_axis) - np.eye(3)
                    
                    axis_aligned_pcd = copy.deepcopy(z_axis_aligned_pcd)
                    axis_aligned_pcd = axis_aligned_pcd.translate(-aligned_center)
                    
                    transform_matrix = np.eye(4)
                    transform_matrix[:3, :3] = rotation_matrix
                    axis_aligned_pcd = axis_aligned_pcd.transform(transform_matrix)
                    
                    axis_aligned_pcd = axis_aligned_pcd.translate(aligned_center)
                    
                    print("SUCCESS: 180-degree axis alignment applied")
                    return axis_aligned_pcd
                else:
                    print("Axes already parallel, no rotation needed")
                    return z_axis_aligned_pcd
            
        except Exception as e:
            print(f"ERROR in rotation calculation: {e}")
            print("Using z-axis alignment without rotation")
            return z_axis_aligned_pcd
        
    #edited

    
    def apply_icp_refinement(self, reference_pcd: o3d.geometry.PointCloud,
                           axis_aligned_pcd: o3d.geometry.PointCloud,
                           max_iterations: int = 100,
                           tolerance: float = 1e-6) -> o3d.geometry.PointCloud:
        """Apply ICP refinement."""
        print(f"\nApplying ICP refinement (FIXED for Open3D 0.19.0)...")
        print(f"Max iterations: {max_iterations}, Tolerance: {tolerance}")
        
        try:
            threshold = 0.02
            trans_init = np.identity(4)
            
            reg_p2p = o3d.pipelines.registration.registration_icp(
                axis_aligned_pcd, reference_pcd, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
            )
            
            print(f"ICP fitness score: {reg_p2p.fitness:.4f}")
            print(f"ICP inlier RMSE: {reg_p2p.inlier_rmse:.4f}mm")
            print(f"ICP correspondences: {len(reg_p2p.correspondence_set):,}")
            
            icp_aligned_pcd = copy.deepcopy(axis_aligned_pcd)
            icp_aligned_pcd.transform(reg_p2p.transformation)
            
            print("SUCCESS: ICP refinement applied")
            return icp_aligned_pcd
                
        except Exception as e:
            print(f"WARNING: ICP failed ({e}), using axis alignment")
            return axis_aligned_pcd
    
    def apply_complete_alignment(self, reference_pcd: o3d.geometry.PointCloud,
                               scaled_inner_pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Apply complete alignment pipeline: Center -> Z-axis -> PCA -> ICP."""
        print("\n" + "="*60)
        print("APPLYING COMPLETE ALIGNMENT PIPELINE")
        print("="*60)
        
        center_aligned = self.apply_center_alignment(reference_pcd, scaled_inner_pcd)
        z_axis_aligned = self.apply_z_axis_alignment(reference_pcd, center_aligned)
        z_axis_aligned = manual_gap_xyz_alignment(reference_pcd, z_axis_aligned)
        axis_aligned = self.apply_pca_axis_alignment(reference_pcd, z_axis_aligned)
        final_aligned = self.apply_icp_refinement(reference_pcd, axis_aligned)
        
        ref_center = reference_pcd.get_center()
        final_center = final_aligned.get_center()
        center_accuracy = np.linalg.norm(final_center - ref_center)
        
        print(f"\nCOMPLETE ALIGNMENT SUMMARY:")
        print(f"Final center accuracy: {center_accuracy:.3f}mm")
        
        if center_accuracy < 50:
            alignment_quality = "GOOD"
        elif center_accuracy < 200:
            alignment_quality = "ACCEPTABLE"
        else:
            alignment_quality = "NEEDS_IMPROVEMENT"
        
        print(f"Alignment quality: {alignment_quality}")
        print("="*60)

        #edited
        """final_center = final_aligned.get_center()
        ref_center = reference_pcd.get_center()
        offset = ref_center - final_center
        if np.linalg.norm(offset) > 1e-3:  # More precise: less than 1 mm error allowed
            print(f"Applying final center correction: {offset}")
            final_aligned = final_aligned.translate(offset)
            # CHECK again: now centers should be identical (within floating-point limits)!
            print(f"New center after correction: {final_aligned.get_center()} vs ref {ref_center}")"""
        lower_by = 21.75
        final_aligned.translate([0, 0, -lower_by])

        return final_aligned
    
    def apply_complete_alignment_with_fixed_scaling(self, reference_pcd: o3d.geometry.PointCloud,
                                                   cleaned_inner_pcd: o3d.geometry.PointCloud,
                                                   scan_identifier: str = "unknown") -> Tuple[o3d.geometry.PointCloud, float]:
        """Apply complete alignment pipeline with optional fixed scaling."""
        print(f"\n{'='*80}")
        print(f"ENHANCED ALIGNMENT PIPELINE WITH FIXED SCALING - {scan_identifier}")
        print(f"{'='*80}")
        
        # Step 1: Apply scaling (fixed or automatic)
        if self.enable_fixed_scaling:
            print(f"\nStep 1: Applying FIXED scaling...")
            scaled_pcd, scale_factor = self.apply_fixed_scaling(cleaned_inner_pcd, scan_identifier)
        else:
            print(f"\nStep 1: Using automatic scaling (compatibility mode)...")
            scaled_pcd = copy.deepcopy(cleaned_inner_pcd)
            scale_factor = 1.0
        
        # Step 2: Apply complete alignment pipeline
        print(f"\nStep 2: Applying complete alignment pipeline...")
        aligned_pcd = self.apply_complete_alignment(reference_pcd, scaled_pcd)
        
        # Step 3: Store scan data for potential post-processing ICP
        self.dual_scan_data[scan_identifier] = {
            'aligned_pcd': copy.deepcopy(aligned_pcd),
            'scale_factor': scale_factor,
            'scan_name': scan_identifier
        }
        
        print(f"\nENHANCED ALIGNMENT COMPLETED:")
        print(f"  Scan: {scan_identifier}")
        print(f"  Scale factor: {scale_factor}")
        print(f"  Fixed scaling enabled: {self.enable_fixed_scaling}")
        print(f"  Stored for dual-scan processing: YES")
        print(f"{'='*80}")
        
        return aligned_pcd, scale_factor
    
    def apply_post_processing_icp_if_ready(self) -> bool:
        """
        Automatically apply post-processing ICP if two scans are ready.
        This is called automatically when the second scan is processed.
        
        Returns:
            True if post-processing ICP was applied, False otherwise
        """
        if len(self.dual_scan_data) < 2:
            return False
        
        # Get the two scans
        scan_names = list(self.dual_scan_data.keys())
        scan1_name = scan_names[0]
        scan2_name = scan_names[1]
        
        scan1_pcd = self.dual_scan_data[scan1_name]['aligned_pcd']
        scan2_pcd = self.dual_scan_data[scan2_name]['aligned_pcd']
        
        # Apply post-processing ICP
        print(f"\n{'='*80}")
        print("AUTO-APPLYING POST-PROCESSING ICP BETWEEN DUAL SCANS")
        print(f"{'='*80}")
        
        scan1_refined, scan2_refined, icp_results = self._apply_post_processing_icp(
            scan1_pcd, scan2_pcd, scan1_name, scan2_name
        )
        
        # Update stored data with refined alignment
        self.dual_scan_data[scan1_name]['aligned_pcd'] = scan1_refined
        self.dual_scan_data[scan2_name]['aligned_pcd'] = scan2_refined
        self.post_icp_results = icp_results
        
        # Log results
        print(f"\nPOST-PROCESSING ICP AUTO-APPLIED:")
        print(f"  Improvement: {icp_results.get('improvement_mm', 0):.1f}mm")
        print(f"  Quality: {icp_results.get('quality', 'UNKNOWN')}")
        print(f"  Final separation: {icp_results.get('final_separation', 0):.1f}mm")
        print(f"{'='*80}")
        
        return True
    
    def _apply_post_processing_icp(self, scan1_pcd: o3d.geometry.PointCloud, 
                                  scan2_pcd: o3d.geometry.PointCloud,
                                  scan1_name: str = "scan1",
                                  scan2_name: str = "scan2",
                                  max_iterations: int = 100) -> Tuple[o3d.geometry.PointCloud, 
                                                                     o3d.geometry.PointCloud, 
                                                                     Dict[str, Any]]:
        """Apply post-processing ICP between two inner scans."""
        print(f"POST-PROCESSING ICP: Aligning {scan2_name} to {scan1_name} using multi-stage ICP...")
        
        scan1_count = len(scan1_pcd.points)
        scan2_count = len(scan2_pcd.points)
        
        if scan1_count == 0 or scan2_count == 0:
            return scan1_pcd, scan2_pcd, {"success": False, "error": "Empty point clouds"}
        
        print(f"  {scan1_name}: {scan1_count:,} points (reference)")
        print(f"  {scan2_name}: {scan2_count:,} points (to be aligned)")
        
        # Initial center alignment
        scan1_center = scan1_pcd.get_center()
        scan2_center = scan2_pcd.get_center()
        initial_offset = scan1_center - scan2_center
        initial_distance = np.linalg.norm(initial_offset)
        
        print(f"  Initial center separation: {initial_distance:.1f}mm")
        
        initial_transform = np.eye(4)
        initial_transform[0:3, 3] = initial_offset
        
        scan2_aligned = copy.deepcopy(scan2_pcd)
        scan2_aligned.transform(initial_transform)
        
        aligned_center = scan2_aligned.get_center()
        center_error = np.linalg.norm(scan1_center - aligned_center)
        print(f"  After center alignment: {center_error:.3f}mm separation")
        
        # Multi-threshold ICP
        thresholds = [500.0, 200.0, 100.0, 50.0, 25.0]
        icp_results = {
            "success": True,
            "initial_separation": initial_distance,
            "center_alignment_error": center_error,
            "icp_stages": [],
            "final_fitness": 0.0,
            "final_rmse": float('inf'),
            "total_correspondences": 0,
            "final_separation": 0.0
        }
        
        for i, threshold in enumerate(thresholds):
            try:
                estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
                criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-8,
                    relative_rmse=1e-8,
                    max_iteration=max_iterations
                )
                
                reg_result = o3d.pipelines.registration.registration_icp(
                    scan2_aligned, scan1_pcd, threshold,
                    estimation_method=estimation,
                    criteria=criteria
                )
                
                scan2_aligned.transform(reg_result.transformation)
                
                stage_result = {
                    "threshold": threshold,
                    "fitness": reg_result.fitness,
                    "inlier_rmse": reg_result.inlier_rmse,
                    "correspondences": len(reg_result.correspondence_set)
                }
                icp_results["icp_stages"].append(stage_result)
                
                print(f"    Stage {i+1} (threshold={threshold:.0f}mm): Fitness={reg_result.fitness:.4f}, RMSE={reg_result.inlier_rmse:.2f}mm, Correspondences={len(reg_result.correspondence_set):,}")
                
                if reg_result.fitness > icp_results["final_fitness"]:
                    icp_results["final_fitness"] = reg_result.fitness
                    icp_results["final_rmse"] = reg_result.inlier_rmse
                    icp_results["total_correspondences"] = len(reg_result.correspondence_set)
                
                if reg_result.fitness > 0.8 and reg_result.inlier_rmse < 10.0:
                    print(f"    Excellent alignment achieved, stopping early")
                    break
                    
            except Exception as e:
                print(f"    ERROR in ICP stage {i+1}: {str(e)}")
                continue
        
        # Final assessment
        final_scan1_center = scan1_pcd.get_center()
        final_scan2_center = scan2_aligned.get_center()
        final_separation = np.linalg.norm(final_scan1_center - final_scan2_center)
        icp_results["final_separation"] = final_separation
        
        improvement = initial_distance - final_separation
        improvement_pct = (improvement / initial_distance) * 100 if initial_distance > 0 else 0
        
        # Quality assessment
        if icp_results["final_fitness"] > 0.7 and final_separation < 50:
            quality = "EXCELLENT"
        elif icp_results["final_fitness"] > 0.3 and final_separation < 200:
            quality = "GOOD"
        else:
            quality = "NEEDS_IMPROVEMENT"
        
        icp_results["quality"] = quality
        icp_results["improvement_mm"] = improvement
        icp_results["improvement_percent"] = improvement_pct
        
        print(f"  Final separation: {final_separation:.1f}mm")
        print(f"  Improvement: {improvement:.1f}mm ({improvement_pct:.1f}%)")
        print(f"  Quality: {quality}")
        
        return scan1_pcd, scan2_aligned, icp_results
    
    def get_dual_scan_results(self) -> Dict[str, Any]:
        """Get results from dual-scan processing including post-ICP results."""
        results = {
            'dual_scan_data': self.dual_scan_data,
            'post_icp_applied': self.post_icp_results is not None,
            'post_icp_results': self.post_icp_results,
            'scaling_summary': self.get_scaling_summary()
        }
        
        if len(self.dual_scan_data) >= 2:
            scan_names = list(self.dual_scan_data.keys())
            results['scan1_name'] = scan_names[0]
            results['scan2_name'] = scan_names[1]
            results['scan1_aligned'] = self.dual_scan_data[scan_names[0]]['aligned_pcd']
            results['scan2_aligned'] = self.dual_scan_data[scan_names[1]]['aligned_pcd']
        
        return results
    
    def get_scaling_summary(self) -> Dict[str, Any]:
        """Get summary of all scaling operations performed."""
        return {
            'fixed_scaling_enabled': self.enable_fixed_scaling,
            'fixed_scale_factor': self.fixed_scale_factor,
            'scaling_results': self.scaling_results,
            'total_scans_processed': len(self.scaling_results)
        }
    
    def reset_dual_scan_data(self):
        """Reset dual-scan data (useful for new processing sessions)."""
        self.dual_scan_data = {}
        self.post_icp_results = None
        self.scaling_results = {}


if __name__ == "__main__":
    print("=" * 80)
    print("TESTING COMPLETE ENHANCED ALIGNMENT MODULE")
    print("=" * 80)
    
    processor = AlignmentProcessor(enable_fixed_scaling=True, fixed_scale_factor=1000.0)
    print(f"Complete enhanced alignment processor created")
    print(f"  Fixed scaling: {processor.enable_fixed_scaling}")
    print(f"  Scale factor: {processor.fixed_scale_factor}")
    
    print("\nCOMPLETE ENHANCED FEATURES:")
    print("- Fixed scaling (eliminates scale discrepancies)")
    print("- Automatic post-processing ICP between dual scans")
    print("- Built-in dual-scan data management")
    print("- Enhanced logging and tracking")
    print("- Complete backward compatibility")
    
    print("\nAUTO-FEATURES:")
    print("- Automatically applies post-ICP when 2nd scan is processed")
    print("- Stores scan data for seamless dual-scan workflow")
    print("- Provides comprehensive results summary")
    print("- No manual intervention required")
    
    print("\nUSAGE PATTERNS:")
    print("1. Single scan: Use apply_complete_alignment_with_fixed_scaling()")
    print("2. Dual scan: Process scan1, then scan2 - post-ICP applied automatically")
    print("3. Get results: Use get_dual_scan_results() for complete summary")
    
    print("=" * 80)
