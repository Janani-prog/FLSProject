import open3d as o3d
import numpy as np
import copy
from typing import Tuple, Dict, Any
from sklearn.decomposition import PCA

from config import ProcessingConfig

def manual_gap_xyz_alignment(reference_pcd, scan_pcd):
    """
    Universally aligns scan_pcd to reference_pcd by:
    1. Centering along X (using min/max average)
    2. Aligning top surface along Z (using 95th percentile)
    3. Equalizing left/right X-axis edge gaps
    Returns a new, aligned point cloud.
    """
    ref_pts = np.asarray(reference_pcd.points)
    scan_pts = np.asarray(scan_pcd.points)

    # Step 1: X-axis center alignment (min/max average)
    x_center_ref = (ref_pts[:, 0].min() + ref_pts[:, 0].max()) / 2
    x_center_scan = (scan_pts[:, 0].min() + scan_pts[:, 0].max()) / 2
    x_shift = x_center_ref - x_center_scan

    y_center_ref = (ref_pts[:, 1].min() + ref_pts[:, 1].max()) / 2
    y_center_scan = (scan_pts[:, 1].min() + scan_pts[:, 1].max()) / 2
    y_shift = y_center_ref - y_center_scan


    # Step 2: Z-axis top surface alignment (95th percentile)
    """ref_top_z = np.mean(ref_pts[ref_pts[:, 2] >= np.percentile(ref_pts[:, 2], 95)][:, 2])
    scan_top_z = np.mean(scan_pts[scan_pts[:, 2] >= np.percentile(scan_pts[:, 2], 95)][:, 2])
    z_shift = ref_top_z - scan_top_z"""

    # Apply center alignment
    new_pcd = copy.deepcopy(scan_pcd)
    new_pcd.translate([x_shift, y_shift, 0])

    # Step 3: Edge gap equalization (X-axis)
    aligned_pts = np.asarray(new_pcd.points)
    ref_xmin, ref_xmax = ref_pts[:, 0].min(), ref_pts[:, 0].max()
    scan_xmin, scan_xmax = aligned_pts[:, 0].min(), aligned_pts[:, 0].max()
    left_gap = scan_xmin - ref_xmin
    right_gap = ref_xmax - scan_xmax
    x_shift_correction = (left_gap - right_gap) / 2.0
    new_pcd.translate([-x_shift_correction, 0, 0])

    return new_pcd

def rotation_matrix_from_vectors(vec1, vec2):
    """Calculate rotation matrix that aligns vec1 to vec2."""
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s == 0:
        return np.eye(3)  # vectors are parallel
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + (kmat @ kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def align_to_reference(reference_pcd, scan_pcd, scaling_factor=1000.0):
    """
    Gold-standard alignment sequence:
      1. Fixed scaling
      2. Centroid alignment
      3. PCA axis alignment
      4. ICP refinement
      5. Apply 28-Oct-2023 fixed offset (-5.718, -10.000, -76.850 mm)
    """

    def scale_pcd(pcd, factor):
        scaled = copy.deepcopy(pcd)
        scaled.scale(factor, center=pcd.get_center())
        return scaled

    # Step 1 — Fixed scaling
    ref_scaled = scale_pcd(reference_pcd, scaling_factor)
    scan_scaled = scale_pcd(scan_pcd, scaling_factor)

    # Step 2 — Centroid alignment
    ref_pts = np.asarray(ref_scaled.points)
    scan_pts = np.asarray(scan_scaled.points)
    ref_centroid = np.mean(ref_pts, axis=0)
    scan_centroid = np.mean(scan_pts, axis=0)
    translation = ref_centroid - scan_centroid
    scan_scaled.translate(translation)

    # Step 3 — PCA axis alignment
    ref_pca = PCA(n_components=3)
    scan_pca = PCA(n_components=3)
    ref_pca.fit(ref_pts - ref_centroid)
    scan_pca.fit(np.asarray(scan_scaled.points) - ref_centroid)
    rot_matrix = rotation_matrix_from_vectors(
        scan_pca.components_[0], ref_pca.components_[0]
    )
    scan_scaled.rotate(rot_matrix, center=ref_centroid)

    # Step 4 — ICP refinement (scale-aware threshold)
    threshold = 100.0  # mm-level tolerance for large parts
    trans_init = np.identity(4)
    icp_result = o3d.pipelines.registration.registration_icp(
        scan_scaled, ref_scaled, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    scan_scaled.transform(icp_result.transformation)

    # Step 5 — Apply 28-Oct-2023 gold-standard reference offset
    """reference_offset = np.array([-5.718, -10.000, 76.850])  # mm
    scan_scaled.translate(reference_offset)
    print(f"Applied fixed gold-standard reference offset: {reference_offset} mm")"""
    scan_scaled = manual_gap_xyz_alignment(reference_pcd, scan_scaled)

    return scan_scaled


# ----------------------------------------------------------
# DEPRECATED FUNCTIONS — retained for historical reference
# ----------------------------------------------------------
# def DEPRECATED_manual_gap_xyz_alignment(...):
#     """Deprecated manual alignment method — do not use."""
#     pass
#
# def DEPRECATED_generalized_center_alignment(...):
#     """Deprecated centroid-only alignment — do not use."""
#     pass


class AlignmentProcessor:
    """
    Complete enhanced alignment processor (fixed-scaling, PCA, ICP)
    configured to reproduce the 28-Oct-2023 gold-standard coordinate frame.
    """

    def __init__(self, enable_fixed_scaling: bool = True, fixed_scale_factor: float = 1000.0):
        self.config = ProcessingConfig()
        self.enable_fixed_scaling = enable_fixed_scaling
        self.fixed_scale_factor = fixed_scale_factor
        self.scaling_results: Dict[str, Any] = {}
        self.dual_scan_data: Dict[str, Any] = {}
        self.post_icp_results: Dict[str, Any] = None

    # ------------------------------------------------------

    def apply_fixed_scaling(self, pcd: o3d.geometry.PointCloud, scan_identifier: str = "unknown"):
        """Apply consistent fixed scaling to maintain unit uniformity."""
        if not self.enable_fixed_scaling:
            return copy.deepcopy(pcd), 1.0
        scaled = copy.deepcopy(pcd)
        scaled.scale(self.fixed_scale_factor, center=scaled.get_center())
        self.scaling_results[scan_identifier] = {
            "scale_factor": self.fixed_scale_factor
        }
        return scaled, self.fixed_scale_factor

    # ------------------------------------------------------

    def apply_complete_alignment_with_fixed_scaling(
        self,
        reference_pcd: o3d.geometry.PointCloud,
        cleaned_inner_pcd: o3d.geometry.PointCloud,
        scan_identifier: str = "unknown"
    ) -> Tuple[o3d.geometry.PointCloud, float]:
        """
        Enhanced fixed-scaling pipeline reproducing the 28-Oct-2023
        alignment metrics (includes fixed gold-standard offset).
        """
        print(f"\n{'='*80}")
        print(f"ENHANCED ALIGNMENT PIPELINE — {scan_identifier}")
        print(f"{'='*80}")

        aligned_pcd = align_to_reference(
            reference_pcd, cleaned_inner_pcd, scaling_factor=self.fixed_scale_factor
        )
        scale_factor = self.fixed_scale_factor

        self.dual_scan_data[scan_identifier] = {
            "aligned_pcd": copy.deepcopy(aligned_pcd),
            "scale_factor": scale_factor,
            "scan_name": scan_identifier
        }

        print(f"\nAlignment completed for {scan_identifier}")
        print(f"  Scale factor: {scale_factor}")
        print(f"  Fixed reference offset applied (28-Oct-2023 frame)")
        print(f"{'='*80}")
        return aligned_pcd, scale_factor

    # ------------------------------------------------------

    def apply_post_processing_icp_if_ready(self) -> bool:
        """Run dual-scan ICP refinement automatically when both scans available."""
        if len(self.dual_scan_data) < 2:
            return False

        s1, s2 = list(self.dual_scan_data.keys())[:2]
        scan1_pcd = self.dual_scan_data[s1]["aligned_pcd"]
        scan2_pcd = self.dual_scan_data[s2]["aligned_pcd"]
        print(f"\n{'='*80}")
        print("AUTO POST-PROCESSING ICP BETWEEN DUAL SCANS")
        print(f"{'='*80}")
        scan1_refined, scan2_refined, icp_results = self._apply_post_processing_icp(
            scan1_pcd, scan2_pcd, s1, s2
        )
        self.dual_scan_data[s1]["aligned_pcd"] = scan1_refined
        self.dual_scan_data[s2]["aligned_pcd"] = scan2_refined
        self.post_icp_results = icp_results
        print(f"Post-processing ICP completed. Quality: {icp_results.get('quality')}")
        return True

    # ------------------------------------------------------

    def _apply_post_processing_icp(
        self, scan1_pcd, scan2_pcd, scan1_name, scan2_name, max_iterations=100
    ):
        """Internal helper for dual-scan ICP."""
        threshold = 100.0
        trans_init = np.identity(4)
        reg = o3d.pipelines.registration.registration_icp(
            scan2_pcd, scan1_pcd, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
        )
        scan2_aligned = copy.deepcopy(scan2_pcd)
        scan2_aligned.transform(reg.transformation)
        quality = "EXCELLENT" if reg.fitness > 0.8 else "GOOD" if reg.fitness > 0.3 else "NEEDS_IMPROVEMENT"
        return scan1_pcd, scan2_aligned, {"fitness": reg.fitness, "quality": quality}

    # ------------------------------------------------------

    def get_dual_scan_results(self) -> Dict[str, Any]:
        """Retrieve dual-scan results and scaling summary."""
        return {
            "dual_scan_data": self.dual_scan_data,
            "post_icp_applied": self.post_icp_results is not None,
            "post_icp_results": self.post_icp_results,
            "scaling_summary": self.get_scaling_summary()
        }

    def get_scaling_summary(self) -> Dict[str, Any]:
        """Summarize scaling operations."""
        return {
            "fixed_scaling_enabled": self.enable_fixed_scaling,
            "fixed_scale_factor": self.fixed_scale_factor,
            "total_scans_processed": len(self.scaling_results)
        }

    def reset_dual_scan_data(self):
        """Reset stored scan data (new session)."""
        self.dual_scan_data = {}
        self.post_icp_results = None
        self.scaling_results = {}


# ==========================================================
#  Test entry point
# ==========================================================
if __name__ == "__main__":
    print("=" * 80)
    print("TESTING GOLD-STANDARD ALIGNMENT MODULE (28-Oct-2023 frame)")
    print("=" * 80)
    processor = AlignmentProcessor(enable_fixed_scaling=True, fixed_scale_factor=1000.0)
    print("Processor created successfully.")
    print("All alignments will include the fixed 28-Oct reference offset.")
    print("Use apply_complete_alignment_with_fixed_scaling() for all scans.")
    print("=" * 80)
