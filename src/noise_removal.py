"""
Fixed Safe Noise Removal Processor - Proper Bottom Plate Removal
Restores original V3 plane removal functionality with safety limits
CORRECTED: Maintains original noise removal effectiveness while improving point retention
"""


import open3d as o3d
import numpy as np
import gc
from typing import Optional, Tuple
from pathlib import Path




class NoiseRemovalProcessor:
    """Safe noise removal processor with proper bottom plate removal."""
   
    def __init__(self):
        """Initialize with conservative but effective memory limits."""
        # Memory estimates for 32GB system
        self.total_ram = 32 * (1024**3)  # 32GB in bytes
        self.windows_overhead = 0.45  # 45% Windows + other apps
        self.safety_margin = 0.20     # Reduced safety margin from 25% to 20%
        self.usable_ram = self.total_ram * (1 - self.windows_overhead - self.safety_margin)
       
        # More realistic point cloud memory estimation
        self.bytes_per_point = 120  # Reduced from 150 for better memory estimate
       
        # INCREASED safe points limit to retain more data
        self.max_safe_points = min(
            int(self.usable_ram / self.bytes_per_point),
            15_000_000  # INCREASED from 8M to 15M points for better detail retention
        )
       
        print(f"SAFE Memory Analysis:")
        print(f"  Total RAM: {self.total_ram / (1024**3):.1f} GB")
        print(f"  Usable for processing: {self.usable_ram / (1024**3):.1f} GB")
        print(f"  Bytes per point: {self.bytes_per_point}")
        print(f"  Safe limit: {self.max_safe_points:,} points MAX")


    def apply_v3_noise_removal(self, point_cloud: o3d.geometry.PointCloud) -> Optional[o3d.geometry.PointCloud]:
        """
        Apply V3 noise removal with PROPER bottom plate removal.
        CORRECTED: Uses original noise removal effectiveness with improved initial downsampling
       
        Args:
            point_cloud: Input point cloud
           
        Returns:
            Safely processed point cloud with bottom plate removed
        """
        if point_cloud is None or len(point_cloud.points) == 0:
            print("ERROR: Invalid input point cloud")
            return None
           
        try:
            print(f"\n=== V3 NOISE REMOVAL WITH IMPROVED RETENTION ===")
            input_points = len(point_cloud.points)
            print(f"Input points: {input_points:,}")
            print(f"Target retention: 10-12M points ({(10_000_000/input_points)*100:.1f}-{(12_000_000/input_points)*100:.1f}%)")
           
            # Step 1: IMPROVED initial downsampling (less aggressive than original)
            if input_points > self.max_safe_points:
                print(f"Input exceeds safe limit, applying IMPROVED initial downsampling")
                working_pcd = self._improved_initial_downsample(point_cloud)
            else:
                print(f"Input within safe limits")
                working_pcd = point_cloud
               
            print(f"Working with: {len(working_pcd.points):,} points")
           
            # Force garbage collection
            gc.collect()
           
            # Step 2: CRITICAL - Bottom plate removal (RESTORED FROM ORIGINAL)
            print("Applying BOTTOM PLATE REMOVAL (V3 original method)...")
            working_pcd = self._remove_bottom_plate_v3_original(working_pcd)
           
            # Step 3: ORIGINAL STRENGTH statistical outlier removal (NOT gentler)
            print("Applying statistical outlier removal (original strength)...")
            working_pcd = self._apply_original_statistical_removal(working_pcd)
           
            # Step 4: SELECTIVE DBSCAN clustering (keep only significant clusters)
            print("Applying SELECTIVE V3 DBSCAN clustering...")
            working_pcd = self._apply_selective_v3_dbscan_clustering(working_pcd)
           
            # Final results
            final_points = len(working_pcd.points)
            retention_rate = (final_points / input_points) * 100
           
            print(f"\nCORRECTED V3 Processing Results:")
            print(f"  Input points: {input_points:,}")
            print(f"  Output points: {final_points:,}")
            print(f"  Overall retention: {retention_rate:.1f}%")
           
            if final_points >= 10_000_000:
                print(f"  TARGET ACHIEVED: {final_points:,} points retained (target: 10-12M)")
            elif final_points >= 8_000_000:
                print(f"  CLOSE TO TARGET: {final_points:,} points retained (target: 10-12M)")
            else:
                print(f"  BELOW TARGET: {final_points:,} points retained (target: 10-12M)")
           
            # Final cleanup
            gc.collect()
           
            return working_pcd
           
        except Exception as e:
            print(f"ERROR in V3 noise removal: {str(e)}")
            gc.collect()
            return None


    def _improved_initial_downsample(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        IMPROVED initial downsampling that starts with more points for better final retention.
        Uses smaller voxels but aims for higher intermediate point count.
       
        Args:
            pcd: Input point cloud  
           
        Returns:
            Downsampled point cloud with better initial retention for subsequent processing
        """
        input_points = len(pcd.points)
        # Target MORE points initially so final result has 10-12M after noise removal
        target_points = min(self.max_safe_points, 16_000_000)  # Start with more points
       
        print(f"  IMPROVED initial downsampling: {input_points:,} -> targeting {target_points:,}")
       
        # Use smaller voxel sizes to retain more detail
        voxel_sizes = [0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.012]
       
        for voxel_size in voxel_sizes:
            try:
                temp_pcd = pcd.voxel_down_sample(voxel_size)
                temp_points = len(temp_pcd.points)
               
                print(f"  Testing voxel {voxel_size:.3f}: {temp_points:,} points")
               
                if temp_points <= target_points:
                    retention_percent = (temp_points/input_points)*100
                    print(f"  Using voxel size {voxel_size:.3f}")
                    print(f"  Achieved: {temp_points:,} points ({retention_percent:.1f}% initial retention)")
                    return temp_pcd
                   
            except Exception as e:
                print(f"  Failed voxel {voxel_size}: {e}")
                continue
       
        # Fallback to last voxel size
        final_pcd = pcd.voxel_down_sample(0.012)
        final_points = len(final_pcd.points)
        retention_percent = (final_points/input_points)*100
        print(f"  Fallback: {final_points:,} points ({retention_percent:.1f}% retention)")
       
        return final_pcd


    def _remove_bottom_plate_v3_original(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        RESTORE ORIGINAL V3 bottom plate removal functionality.
        This is the critical step that was missing!
        Args:
            pcd: Input point cloud
        Returns:
            Point cloud with bottom plate removed
        """
        input_points = len(pcd.points)
        try:
            print(f" V3 Bottom Plate Removal - Input: {input_points:,} points")
            points = np.asarray(pcd.points)
            # Original V3 method: Remove bottom 10% of points by Z-coordinate
            z_coords = points[:, 2]
            z_min = np.min(z_coords)
            z_max = np.max(z_coords)
            z_range = z_max - z_min
            # Remove bottom 10% (this removes the flat plate)
            bottom_threshold = z_min + (z_range * 0.10)  # Bottom 10%
            print(f" Z-range: {z_min:.2f} to {z_max:.2f} (range: {z_range:.2f})")
            print(f" Bottom threshold: {bottom_threshold:.2f}")
            # Keep points above the threshold
            keep_mask = z_coords > bottom_threshold
            # Apply the mask
            filtered_points = points[keep_mask]
            # Create new point cloud
            clean_pcd = o3d.geometry.PointCloud()
            clean_pcd.points = o3d.utility.Vector3dVector(filtered_points)
            # Copy colors if available
            if pcd.has_colors():
                colors = np.asarray(pcd.colors)
                filtered_colors = colors[keep_mask]
                clean_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
            # Copy normals if available
            if pcd.has_normals():
                normals = np.asarray(pcd.normals)
                filtered_normals = normals[keep_mask]
                clean_pcd.normals = o3d.utility.Vector3dVector(filtered_normals)
            removed_points = input_points - len(clean_pcd.points)
            removal_rate = (removed_points / input_points) * 100
            print(f" Bottom plate removal: {removed_points:,} points removed ({removal_rate:.1f}%)")
            print(f" Remaining: {len(clean_pcd.points):,} points")
            return clean_pcd
        except Exception as e:
            print(f" Bottom plate removal failed: {e}")
            return pcd


    def _apply_original_statistical_removal(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Apply ORIGINAL STRENGTH statistical outlier removal.
        CORRECTED: Uses original parameters for effective noise removal
        """
        input_points = len(pcd.points)
        nb_neighbors = 20  # Original V3 value - DO NOT reduce
        std_ratio = 2.0    # Original V3 value - DO NOT increase
        try:
            clean_pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )
            removed_points = input_points - len(clean_pcd.points)
            removal_rate = (removed_points / input_points) * 100
            print(f" Statistical removal: {removed_points:,} removed ({removal_rate:.1f}%)")
            return clean_pcd
        except Exception as e:
            print(f" Statistical removal failed: {e}")
            return pcd




    def _apply_selective_v3_dbscan_clustering(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Apply SELECTIVE DBSCAN clustering - keeps only significant clusters, not tiny noise clusters.
        CORRECTED: More selective about which clusters to keep
        """
        input_points = len(pcd.points)
       
        # Original V3 DBSCAN parameters
        eps = 0.05         # Original V3 epsilon
        min_points = 100   # Original V3 min_points
       
        try:
            print(f"  SELECTIVE V3 DBSCAN - eps: {eps}, min_points: {min_points}")
           
            labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
           
            # Get cluster information
            unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
           
            if len(unique_labels) == 0:
                print(f"  No clusters found, keeping original")
                return pcd
           
            # Sort clusters by size (largest first)
            sorted_indices = np.argsort(counts)[::-1]
            sorted_labels = unique_labels[sorted_indices]
            sorted_counts = counts[sorted_indices]
           
            print(f"  Found {len(unique_labels)} clusters")
           
            # SELECTIVE approach: Only keep clusters that are significant
            # Rule 1: Always keep the largest cluster (main structure)
            # Rule 2: Keep additional clusters only if they're substantial (>1% of largest cluster)
            clusters_to_keep = [sorted_labels[0]]  # Always keep largest
            largest_cluster_size = sorted_counts[0]
            significance_threshold = max(largest_cluster_size * 0.01, 1000)  # 1% of largest or min 1000 points
           
            print(f"  Largest cluster: {largest_cluster_size:,} points")
            print(f"  Significance threshold: {significance_threshold:,.0f} points")
           
            total_kept_points = largest_cluster_size
           
            # Check additional clusters for significance
            for i in range(1, len(sorted_labels)):
                cluster_size = sorted_counts[i]
                if cluster_size >= significance_threshold:
                    clusters_to_keep.append(sorted_labels[i])
                    total_kept_points += cluster_size
                    print(f"  Keeping significant cluster {i+1}: {cluster_size:,} points")
                else:
                    print(f"  Rejecting small cluster {i+1}: {cluster_size:,} points (below threshold)")
           
            print(f"  Total clusters kept: {len(clusters_to_keep)}")
            print(f"  Total points in kept clusters: {total_kept_points:,}")
           
            # Create mask for selected clusters
            cluster_mask = np.isin(labels, clusters_to_keep)
           
            # Extract points from selected clusters
            points = np.asarray(pcd.points)[cluster_mask]
            clean_pcd = o3d.geometry.PointCloud()
            clean_pcd.points = o3d.utility.Vector3dVector(points)
           
            # Copy colors and normals if available
            if pcd.has_colors():
                colors = np.asarray(pcd.colors)[cluster_mask]
                clean_pcd.colors = o3d.utility.Vector3dVector(colors)
               
            if pcd.has_normals():
                normals = np.asarray(pcd.normals)[cluster_mask]
                clean_pcd.normals = o3d.utility.Vector3dVector(normals)
           
            final_kept_points = len(clean_pcd.points)
            removed_points = input_points - final_kept_points
            removal_rate = (removed_points / input_points) * 100
            retention_rate = (final_kept_points / input_points) * 100
           
            print(f"  SELECTIVE V3 DBSCAN: Kept {len(clusters_to_keep)} significant clusters")
            print(f"  Retained: {final_kept_points:,} points ({retention_rate:.1f}%)")
            print(f"  Removed: {removed_points:,} points ({removal_rate:.1f}%)")
           
            return clean_pcd
           
        except Exception as e:
            print(f"  DBSCAN clustering failed: {e}")
            return pcd


    def _smart_downsample(self, pcd: o3d.geometry.PointCloud, target_points: int) -> o3d.geometry.PointCloud:
        """
        Legacy method for backward compatibility - redirects to improved version
        """
        return self._improved_initial_downsample(pcd)


    def apply_legacy_v3_noise_removal(self, point_cloud: o3d.geometry.PointCloud) -> Optional[o3d.geometry.PointCloud]:
        """
        Exact legacy V3 implementation but with CORRECTED parameters.
        CORRECTED: Uses original noise removal effectiveness with improved initial retention
        """
        if point_cloud is None or len(point_cloud.points) == 0:
            return None
           
        try:
            print("Applying CORRECTED legacy V3 noise removal...")
           
            input_points = len(point_cloud.points)
           
            # Safety check with improved limits
            if input_points > self.max_safe_points:
                print(f"Safety downsampling: {input_points:,} -> {self.max_safe_points:,}")
                point_cloud = self._improved_initial_downsample(point_cloud)
           
            # CORRECTED V3 sequence - uses original effective parameters
            # Step 1: Less aggressive initial voxel downsampling
            voxel_size = 0.006  # Smaller voxel (from 0.008) for more initial points
            downsampled = point_cloud.voxel_down_sample(voxel_size)
           
            # Step 2: ORIGINAL STRENGTH statistical outlier removal
            filtered, _ = downsampled.remove_statistical_outlier(
                nb_neighbors=20,   # Original V3 - effective at noise removal
                std_ratio=2.0      # Original V3 - effective at noise removal
            )
           
            # Step 3: ORIGINAL DBSCAN with selective cluster retention
            labels = np.array(filtered.cluster_dbscan(eps=0.05, min_points=100))  # Original parameters
           
            if len(labels) > 0:
                unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
                if len(unique_labels) > 0:
                    # SELECTIVE: Keep largest + any significant clusters (>1% of largest)
                    sorted_indices = np.argsort(counts)[::-1]
                    largest_size = counts[sorted_indices[0]]
                    significance_threshold = largest_size * 0.01
                   
                    clusters_to_keep = []
                    for idx in sorted_indices:
                        if counts[idx] >= significance_threshold:
                            clusters_to_keep.append(unique_labels[idx])
                        else:
                            break
                   
                    if clusters_to_keep:
                        cluster_mask = np.isin(labels, clusters_to_keep)
                    else:
                        # Fallback to largest cluster
                        largest_cluster_label = unique_labels[np.argmax(counts)]
                        cluster_mask = labels == largest_cluster_label
                   
                    points = np.asarray(filtered.points)[cluster_mask]
                    final_pcd = o3d.geometry.PointCloud()
                    final_pcd.points = o3d.utility.Vector3dVector(points)
                   
                    if filtered.has_colors():
                        colors = np.asarray(filtered.colors)[cluster_mask]
                        final_pcd.colors = o3d.utility.Vector3dVector(colors)
                       
                    return final_pcd
                   
            return filtered
           
        except Exception as e:
            print(f"ERROR in corrected legacy V3 noise removal: {str(e)}")
            gc.collect()
            return None




# Backward compatibility function
def apply_v3_noise_removal_legacy(point_cloud: o3d.geometry.PointCloud) -> Optional[o3d.geometry.PointCloud]:
    """Legacy function for backward compatibility."""
    processor = NoiseRemovalProcessor()
    return processor.apply_legacy_v3_noise_removal(point_cloud)




if __name__ == "__main__":
    print("=" * 60)
    print("TESTING CORRECTED V3 NOISE REMOVAL")
    print("=" * 60)
   
    processor = NoiseRemovalProcessor()
    print(f"Ready for CORRECTED V3 processing")
    print(f"Safe limit: {processor.max_safe_points:,} points")
    print(f"Target: Clean 10-12M points from 41M input")

