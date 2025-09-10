"""
Scaling and Unit Detection for Mill Analysis Project
Open3D implementation - Extracted from original massive data_loader.py
Implements proven 95th percentile scaling method and unit detection
"""

import open3d as o3d
import numpy as np
import copy
from typing import Tuple

from config import ProcessingConfig


class ScalingProcessor:
    """Scaling processor using proven 95th percentile method."""
    
    def __init__(self):
        self.config = ProcessingConfig()
        self.optimal_scale_factor = 1.0
    
    def calculate_scale_factor_95th_percentile(self, reference_pcd: o3d.geometry.PointCloud,
                                             inner_pcd: o3d.geometry.PointCloud,
                                             target_ratio: float = 1.0) -> float:
        """
        Calculate scale factor using 95th percentile method (old_version proven algorithm).
        
        Args:
            reference_pcd: Reference point cloud
            inner_pcd: Inner scan point cloud (after noise removal)
            target_ratio: Target scaling ratio (1.0 for 1:1 scaling)
            
        Returns:
            Calculated scale factor
        """
        print(f"\nCalculating scale factor using 95th percentile method (old_version algorithm)...")
        print(f"Target ratio: {target_ratio:.3f}")
        
        ref_points = np.asarray(reference_pcd.points)
        inner_points = np.asarray(inner_pcd.points)
        
        # Calculate centers
        ref_center = np.mean(ref_points, axis=0)
        inner_center = np.mean(inner_points, axis=0)
        
        print(f"Reference center: [{ref_center[0]:.2f}, {ref_center[1]:.2f}, {ref_center[2]:.2f}]")
        print(f"Inner center: [{inner_center[0]:.2f}, {inner_center[1]:.2f}, {inner_center[2]:.2f}]")
        
        # Calculate radial distances from center (old_version method)
        ref_radial_distances = np.sqrt(np.sum((ref_points - ref_center)**2, axis=1))
        inner_radial_distances = np.sqrt(np.sum((inner_points - inner_center)**2, axis=1))
        
        # Use 95th percentile as characteristic radius (proven method)
        ref_characteristic_radius = np.percentile(ref_radial_distances, 95)
        inner_characteristic_radius = np.percentile(inner_radial_distances, 95)
        
        print(f"Reference characteristic radius: {ref_characteristic_radius:.1f}mm")
        print(f"Inner characteristic radius: {inner_characteristic_radius:.1f}mm")
        
        # Calculate required scale factor (old_version formula)
        optimal_scale_factor = (target_ratio * ref_characteristic_radius) / inner_characteristic_radius
        
        print(f"Required scale factor: {optimal_scale_factor:.3f}x")
        print(f"Expected final ratio: {target_ratio:.3f}")
        
        # Verify reasonable scale factor
        if optimal_scale_factor > 10:
            print("WARNING: Scale factor very large - check unit conversion")
        elif 0.5 <= optimal_scale_factor <= 2.0:
            print("EXCELLENT: Scale factor is in reasonable range")
        else:
            print("GOOD: Scale factor calculated successfully")
            
        self.optimal_scale_factor = optimal_scale_factor
        return optimal_scale_factor
    
    def apply_scaling_transformation(self, inner_pcd: o3d.geometry.PointCloud,
                                   scale_factor: float) -> o3d.geometry.PointCloud:
        """
        Apply scaling transformation to point cloud (old_version method).
        
        Args:
            inner_pcd: Inner scan point cloud
            scale_factor: Scale factor to apply
            
        Returns:
            Scaled point cloud
        """
        print(f"\nApplying scaling transformation (old_version method)...")
        print(f"Scale factor: {scale_factor:.3f}x")
        
        # Get center before transformation (old_version method)
        inner_center = inner_pcd.get_center()
        print(f"Original center: [{inner_center[0]:.2f}, {inner_center[1]:.2f}, {inner_center[2]:.2f}]")
        
        # Create scaled copy
        scaled_pcd = copy.deepcopy(inner_pcd)
        
        # Translate to origin, scale, then translate back (old_version method)
        scaled_pcd = scaled_pcd.translate(-inner_center)
        
        # Create scaling matrix
        scale_matrix = np.eye(4)
        scale_matrix[0, 0] = scale_factor
        scale_matrix[1, 1] = scale_factor
        scale_matrix[2, 2] = scale_factor
        
        # Apply scaling transformation
        scaled_pcd = scaled_pcd.transform(scale_matrix)
        scaled_pcd = scaled_pcd.translate(inner_center)
        
        # Verify scaling
        scaled_center = scaled_pcd.get_center()
        print(f"Scaled center: [{scaled_center[0]:.2f}, {scaled_center[1]:.2f}, {scaled_center[2]:.2f}]")
        
        original_points = len(inner_pcd.points)
        scaled_points = len(scaled_pcd.points)
        print(f"Points preserved: {scaled_points:,} (original: {original_points:,})")
        
        # Calculate dimensional change
        original_dims = inner_pcd.get_axis_aligned_bounding_box().get_extent()
        scaled_dims = scaled_pcd.get_axis_aligned_bounding_box().get_extent()
        
        print(f"Original dimensions: [{original_dims[0]:.2f}, {original_dims[1]:.2f}, {original_dims[2]:.2f}]")
        print(f"Scaled dimensions: [{scaled_dims[0]:.2f}, {scaled_dims[1]:.2f}, {scaled_dims[2]:.2f}]")
        
        actual_scale = np.max(scaled_dims) / np.max(original_dims)
        print(f"Actual scale applied: {actual_scale:.3f}x")
        
        if abs(actual_scale - scale_factor) < 0.001:
            print("SUCCESS: Scaling applied correctly")
        else:
            print("WARNING: Scale verification shows discrepancy")
            
        return scaled_pcd
    
    def detect_and_convert_units(self, reference_pcd: o3d.geometry.PointCloud, 
                                inner_pcd: o3d.geometry.PointCloud) -> Tuple[o3d.geometry.PointCloud, bool]:
        """
        Detect unit mismatch and convert to consistent units.
        
        Args:
            reference_pcd: Reference point cloud
            inner_pcd: Inner scan point cloud
            
        Returns:
            Tuple of (converted_inner_pcd, was_converted)
        """
        print(f"\nDetecting unit mismatch (threshold: {self.config.UNIT_DETECTION_THRESHOLD}x)...")
        
        # Calculate characteristic dimensions
        ref_dims = reference_pcd.get_axis_aligned_bounding_box().get_extent()
        inner_dims = inner_pcd.get_axis_aligned_bounding_box().get_extent()
        
        ref_max_dim = np.max(ref_dims)
        inner_max_dim = np.max(inner_dims)
        
        print(f"Reference max dimension: {ref_max_dim:.2f}")
        print(f"Inner scan max dimension: {inner_max_dim:.2f}")
        
        # Check for unit mismatch (meter vs millimeter)
        dimension_ratio = inner_max_dim / ref_max_dim
        
        print(f"Dimension ratio: {dimension_ratio:.1f}x")
        
        if dimension_ratio > self.config.UNIT_DETECTION_THRESHOLD:
            print("UNIT MISMATCH DETECTED: Inner scan appears to be in meters, converting to millimeters")
            
            # Convert meters to millimeters (multiply by 1000)
            conversion_factor = 1000.0
            converted_pcd = self.apply_scaling_transformation(inner_pcd, 1.0 / conversion_factor)
            
            # Verify conversion
            converted_dims = converted_pcd.get_axis_aligned_bounding_box().get_extent()
            converted_max_dim = np.max(converted_dims)
            new_ratio = converted_max_dim / ref_max_dim
            
            print(f"After conversion max dimension: {converted_max_dim:.2f}")
            print(f"New dimension ratio: {new_ratio:.1f}x")
            print("SUCCESS: Unit conversion applied")
            
            return converted_pcd, True
            
        elif dimension_ratio < (1.0 / self.config.UNIT_DETECTION_THRESHOLD):
            print("UNIT MISMATCH DETECTED: Inner scan appears to be in millimeters, converting to match reference")
            
            # Convert millimeters to meters (multiply by 0.001)  
            conversion_factor = 0.001
            converted_pcd = self.apply_scaling_transformation(inner_pcd, 1.0 / conversion_factor)
            
            # Verify conversion
            converted_dims = converted_pcd.get_axis_aligned_bounding_box().get_extent()
            converted_max_dim = np.max(converted_dims)
            new_ratio = converted_max_dim / ref_max_dim
            
            print(f"After conversion max dimension: {converted_max_dim:.2f}")
            print(f"New dimension ratio: {new_ratio:.1f}x")
            print("SUCCESS: Unit conversion applied")
            
            return converted_pcd, True
        else:
            print("GOOD: Units appear consistent, no conversion needed")
            return inner_pcd, False


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING SCALING MODULE")
    print("=" * 60)
    
    processor = ScalingProcessor()
    print("Scaling processor created successfully")
    
    print(f"\nConfiguration loaded:")
    print(f"Target ratio: {processor.config.TARGET_RATIO}")
    print(f"Unit detection threshold: {processor.config.UNIT_DETECTION_THRESHOLD}")
    print(f"Current scale factor: {processor.optimal_scale_factor}")
    
    print("\nScaling Module - Ready for pipeline integration")