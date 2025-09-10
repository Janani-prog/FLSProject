"""
Enhanced Wear Heatmap Generator for Mill Analysis Project
Open3D implementation with proper color-coded distance analysis
Generates comprehensive heatmap visualizations with side view analysis
FIXED: Uses config paths for proper output directory structure
"""

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial import KDTree
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import time

from config import ProcessingConfig, get_file_paths


class WearHeatmapGenerator:
    """Enhanced heatmap generator with proper color-coded wear analysis."""
    
    def __init__(self):
        self.config = ProcessingConfig()
        self.file_paths = get_file_paths()  # Get proper config paths
        self.heatmap_analysis = {}
        
    def generate_enhanced_heatmap(self, scan1_pcd: o3d.geometry.PointCloud, 
                                 scan2_pcd: o3d.geometry.PointCloud) -> Dict[str, Any]:
        """
        Generate enhanced color-coded heatmap comparing two scans.
        
        Args:
            scan1_pcd: First point cloud (reference scan)
            scan2_pcd: Second point cloud (comparison scan)
            
        Returns:
            Heatmap analysis results
        """
        print("Generating enhanced color-coded wear heatmap...")
        start_time = time.time()
        
        # Get points from both scans
        points1 = np.asarray(scan1_pcd.points)
        points2 = np.asarray(scan2_pcd.points)
        
        print(f"Scan 1: {len(points1):,} points")
        print(f"Scan 2: {len(points2):,} points")
        
        # Downsample for performance if needed
        max_points = self.config.HEATMAP_MAX_POINTS
        if len(points1) > max_points:
            indices1 = np.random.choice(len(points1), max_points, replace=False)
            points1 = points1[indices1]
            print(f"Downsampled scan 1 to {len(points1):,} points")
            
        if len(points2) > max_points:
            indices2 = np.random.choice(len(points2), max_points, replace=False)
            points2 = points2[indices2]
            print(f"Downsampled scan 2 to {len(points2):,} points")
        
        # Build KDTree for efficient distance calculation
        print("Building spatial index for distance calculation...")
        tree2 = KDTree(points2)
        
        # Calculate distances from scan1 points to nearest scan2 points
        print("Computing point-to-point distances...")
        distances, _ = tree2.query(points1)
        
        # Calculate statistics
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        percentile_95 = np.percentile(distances, 95)
        
        print(f"Distance statistics:")
        print(f"  Min: {min_dist:.2f}mm")
        print(f"  Max: {max_dist:.2f}mm") 
        print(f"  Mean: {mean_dist:.2f}mm")
        print(f"  Std: {std_dist:.2f}mm")
        print(f"  95th percentile: {percentile_95:.2f}mm")
        
        # Store analysis results
        self.heatmap_analysis = {
            'min_distance': min_dist,
            'max_distance': max_dist,
            'mean_distance': mean_dist,
            'std_distance': std_dist,
            'percentile_95': percentile_95,
            'points_analyzed': len(points1),
            'processing_time': time.time() - start_time
        }
        
        # Generate comprehensive heatmap visualization - FIXED: Use config paths
        self._create_enhanced_heatmap_visualization(points1, points2, distances)
        
        return {
            'heatmap_generated': True,
            'analysis': self.heatmap_analysis
        }
    
    def generate_standard_heatmap(self, scan1_pcd: o3d.geometry.PointCloud, 
                                 scan2_pcd: o3d.geometry.PointCloud) -> Dict[str, Any]:
        """Generate standard heatmap (fallback method)."""
        return self.generate_enhanced_heatmap(scan1_pcd, scan2_pcd)
    
    def _create_enhanced_heatmap_visualization(self, points1: np.ndarray, 
                                         points2: np.ndarray,
                                         distances: np.ndarray) -> None:
        """
        Create enhanced heatmap visualization with better layout and proportions.
        FIXED: Better subplot arrangement and sizing to prevent compression.
        """
        # FIXED: Larger figure with better proportions
        fig = plt.figure(figsize=(24, 16))  # Increased from (20, 12)
        
        # Cap distances for better color visualization
        capped_distances = np.clip(distances, 0, 300)
        
        # Define colormap for wear analysis
        colormap = plt.cm.jet
        norm = mcolors.Normalize(vmin=0, vmax=300)
        
        # MAIN HEATMAP VIEWS (larger subplots)
        # Enhanced Top View Heatmap (X-Y plane) - LARGER
        ax1 = plt.subplot(2, 3, 1)
        scatter1 = ax1.scatter(points1[:, 0], points1[:, 1], c=capped_distances, 
                            cmap=colormap, norm=norm, s=2, alpha=0.8)  # Increased point size
        ax1.set_title('Enhanced Top View Heatmap\n(X-Y plane)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('X (mm)', fontsize=12)
        ax1.set_ylabel('Y (mm)', fontsize=12)
        ax1.set_aspect('equal')
        plt.colorbar(scatter1, ax=ax1, label='Wear (mm)', shrink=0.8)
        
        # ENHANCED SIDE VIEW HEATMAP (Y-Z plane) - LARGER - Key Analysis View
        ax2 = plt.subplot(2, 3, 2)
        scatter2 = ax2.scatter(points1[:, 1], points1[:, 2], c=capped_distances,
                            cmap=colormap, norm=norm, s=2, alpha=0.8)
        ax2.set_title('ENHANCED SIDE VIEW HEATMAP\n(Y-Z plane) - Key Analysis View', 
                    fontsize=14, fontweight='bold')
        ax2.set_xlabel('Y (mm)', fontsize=12)
        ax2.set_ylabel('Z (mm)', fontsize=12)
        ax2.set_aspect('equal')  # This keeps circular shape
        plt.colorbar(scatter2, ax=ax2, label='Wear (mm)', shrink=0.8)
        
        # Enhanced Front View Heatmap (X-Z plane) - LARGER
        ax3 = plt.subplot(2, 3, 3)
        scatter3 = ax3.scatter(points1[:, 0], points1[:, 2], c=capped_distances,
                            cmap=colormap, norm=norm, s=2, alpha=0.8)
        ax3.set_title('Enhanced Front View Heatmap\n(X-Z plane)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('X (mm)', fontsize=12)
        ax3.set_ylabel('Z (mm)', fontsize=12)
        plt.colorbar(scatter3, ax=ax3, label='Wear (mm)', shrink=0.8)
        
        # ANALYSIS VIEWS (bottom row)
        # Enhanced Radial Wear Analysis (Y-Z plane)
        ax4 = plt.subplot(2, 3, 4)
        center_y = np.mean(points1[:, 1])
        center_z = np.mean(points1[:, 2])
        radial_distances = np.sqrt((points1[:, 1] - center_y)**2 + (points1[:, 2] - center_z)**2)
        scatter4 = ax4.scatter(radial_distances, points1[:, 2], c=capped_distances,
                            cmap=colormap, norm=norm, s=1, alpha=0.6)
        ax4.set_title('Enhanced Radial Wear Analysis\n(Y-Z plane distance vs wear)', 
                    fontsize=12, fontweight='bold')
        ax4.set_xlabel('Radial Distance in Y-Z plane')
        ax4.set_ylabel('Z (mm)')
        plt.colorbar(scatter4, ax=ax4, label='Wear (mm)', shrink=0.8)
        
        # Enhanced Angular Wear Analysis (Y-Z plane)
        ax5 = plt.subplot(2, 3, 5)
        angles = np.arctan2(points1[:, 2] - center_z, points1[:, 1] - center_y) * 180 / np.pi
        scatter5 = ax5.scatter(angles, capped_distances, c=capped_distances,
                            cmap=colormap, norm=norm, s=1, alpha=0.6)
        ax5.set_title('Enhanced Angular Wear Analysis\n(Y-Z plane angle vs wear)', 
                    fontsize=12, fontweight='bold')
        ax5.set_xlabel('Angle in Y-Z plane (degrees)')
        ax5.set_ylabel('Wear (mm)')
        plt.colorbar(scatter5, ax=ax5, label='Wear (mm)', shrink=0.8)
        
        # Enhanced Analysis Summary - LARGER TEXT BOX
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = f"""ENHANCED HEATMAP ANALYSIS

    SPATIAL DISTRIBUTION:
    Y range: {np.min(points1[:, 1]):.0f} to {np.max(points1[:, 1]):.0f}
    Z range: {np.min(points1[:, 2]):.0f} to {np.max(points1[:, 2]):.0f}
    Y-Z center: ({center_y:.1f}, {center_z:.1f})

    RADIAL ANALYSIS (Y-Z plane):
    Min radial distance: {np.min(radial_distances):.1f}
    Max radial distance: {np.max(radial_distances):.1f}
    Mean radial distance: {np.mean(radial_distances):.1f}

    WEAR STATISTICS:
    Min wear: {np.min(capped_distances):.1f}mm
    Max wear: {np.max(capped_distances):.1f}mm
    Mean wear: {np.mean(capped_distances):.1f}mm
    Std wear: {np.std(capped_distances):.1f}mm

    CORRELATIONS:
    Angular-wear: {np.corrcoef(angles, capped_distances)[0,1]:.3f}
    Radial-wear: {np.corrcoef(radial_distances, capped_distances)[0,1]:.3f}
    Y-wear: {np.corrcoef(points1[:, 1], capped_distances)[0,1]:.3f}
    Z-wear: {np.corrcoef(points1[:, 2], capped_distances)[0,1]:.3f}

    ENHANCED INSIGHT:
    Y-Z plane provides circular mill cross-section
    Radial patterns show liner wear distribution
    Angular patterns reveal directional wear trends
    Critical for understanding mill dynamics"""
        
        ax6.text(0.02, 0.98, summary_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        # FIXED: Better spacing and layout
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, 
                        wspace=0.3, hspace=0.3)  # More space between subplots
        
        # Use config paths for proper output directory
        output_path = self.file_paths['heatmaps_dir'] / 'enhanced_wear_heatmap_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Enhanced wear heatmap analysis saved: {output_path}")


if __name__ == "__main__":
    print("Enhanced Wear Heatmap Generator - Ready for integration with existing workflow")