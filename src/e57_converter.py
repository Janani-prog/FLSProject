"""
Professional E57 to PLY Converter for Mill Analysis Project
Fixed implementation - prevents duplicate detection and improves efficiency
"""

import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Set
import time
import gc


class E57Converter:
    """Professional E57 to PLY converter with proper error handling and duplicate prevention."""
    
    def __init__(self):
        self.conversion_log = []
        self.processed_files: Set[str] = set()  # Track processed files to prevent duplicates
        self.pye57_available = self._check_pye57_availability()
    
    def _check_pye57_availability(self) -> bool:
        """Check if pye57 library is available."""
        try:
            import pye57
            print(f"pye57 library available - version: {getattr(pye57, '__version__', 'unknown')}")
            return True
        except ImportError:
            print("WARNING: pye57 library not available - E57 support disabled")
            return False
    
    def convert_e57_to_ply(self, e57_path: Path) -> Optional[Path]:
        """
        Convert E57 file to PLY format and store in same directory.
        """
        if not self.pye57_available:
            print("ERROR: pye57 library not available")
            return None
        
        if not e57_path.exists():
            print(f"ERROR: E57 file not found: {e57_path}")
            return None
        
        # Check if already processed to prevent duplicates
        file_key = str(e57_path.absolute())
        if file_key in self.processed_files:
            print(f"File already processed: {e57_path.name}")
            output_path = e57_path.parent / f"{e57_path.stem}_converted.ply"
            return output_path if output_path.exists() else None
        
        print(f"Converting E57 to PLY: {e57_path.name}")
        start_time = time.time()
        
        try:
            import pye57
            
            # Create output path in same directory as E57 file
            output_path = e57_path.parent / f"{e57_path.stem}_converted.ply"
            
            # Check if converted file already exists
            if output_path.exists():
                print(f"Converted file already exists: {output_path.name}")
                self.processed_files.add(file_key)
                return output_path
            
            # Open E57 file
            with pye57.E57(str(e57_path), mode='r') as e57_file:
                
                # Get file information
                print("E57 file info:")
                self._print_e57_info(e57_file)
                
                # Read all point cloud data
                all_points = []
                all_colors = []
                
                # Get scan count - handle both properties and methods
                try:
                    if hasattr(e57_file, 'data3d_count'):
                        scan_count = e57_file.data3d_count
                        if callable(scan_count):
                            scan_count = scan_count()
                    elif hasattr(e57_file, 'scan_count'):
                        scan_count = e57_file.scan_count
                        if callable(scan_count):
                            scan_count = scan_count()
                    else:
                        scan_count = len(list(e57_file.data3d))
                        
                    print(f"Found {scan_count} scan(s) in E57 file")
                    
                except Exception as e:
                    print(f"Could not determine scan count: {e}")
                    scan_count = 1
                
                # Process each scan
                processed_scans = 0
                for scan_index in range(scan_count):
                    try:
                        print(f"Processing scan {scan_index + 1}/{scan_count}...")
                        
                        scan_data = self._read_scan_data(e57_file, scan_index)
                        
                        if scan_data is not None:
                            points, colors = scan_data
                            if len(points) > 0:
                                all_points.append(points)
                                if colors is not None and len(colors) > 0:
                                    all_colors.append(colors)
                                print(f"Scan {scan_index + 1}: {len(points):,} points")
                                processed_scans += 1
                            else:
                                print(f"Scan {scan_index + 1}: No points found")
                        else:
                            print(f"Scan {scan_index + 1}: Failed to read data")
                            
                    except Exception as e:
                        print(f"Error processing scan {scan_index + 1}: {e}")
                        continue
                
                # Fallback approach if no scans processed
                if processed_scans == 0:
                    print("Trying alternative scan reading approach...")
                    try:
                        all_data = self._read_all_scan_data(e57_file)
                        if all_data:
                            points, colors = all_data
                            if len(points) > 0:
                                all_points.append(points)
                                if colors is not None and len(colors) > 0:
                                    all_colors.append(colors)
                                print(f"Alternative method: {len(points):,} points")
                                processed_scans = 1
                    except Exception as e:
                        print(f"Alternative approach also failed: {e}")
                
                if processed_scans == 0:
                    print("ERROR: No scans could be processed")
                    self.processed_files.add(file_key)  # Mark as processed even if failed
                    return None
                
                # Combine all points
                if len(all_points) > 0:
                    combined_points = np.vstack(all_points)
                    combined_colors = np.vstack(all_colors) if len(all_colors) > 0 else None
                    
                    print(f"Total points combined: {len(combined_points):,}")
                    
                    # Create Open3D point cloud
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(combined_points)
                    
                    if combined_colors is not None:
                        if combined_colors.max() > 1.0:
                            combined_colors = combined_colors / 255.0
                        pcd.colors = o3d.utility.Vector3dVector(combined_colors)
                        print("Colors preserved in conversion")
                    
                    # Save to PLY
                    success = o3d.io.write_point_cloud(str(output_path), pcd)
                    
                    if success:
                        conversion_time = time.time() - start_time
                        print(f"Conversion completed in {conversion_time:.2f} seconds")
                        print(f"Converted file saved: {output_path}")
                        
                        # Mark as processed and log conversion
                        self.processed_files.add(file_key)
                        self.conversion_log.append({
                            'original': str(e57_path),
                            'converted': str(output_path),
                            'points': len(combined_points),
                            'has_colors': combined_colors is not None,
                            'conversion_time': conversion_time
                        })
                        
                        return output_path
                    else:
                        print("ERROR: Failed to save PLY file")
                        self.processed_files.add(file_key)
                        return None
                else:
                    print("ERROR: No point data found in E57 file")
                    self.processed_files.add(file_key)
                    return None
                    
        except Exception as e:
            print(f"ERROR converting E57 file: {e}")
            self.processed_files.add(file_key)
            return None
    
    def _print_e57_info(self, e57_file) -> None:
        """Print E57 file information using available API."""
        try:
            if hasattr(e57_file, 'image2d_count'):
                count = e57_file.image2d_count
                if callable(count):
                    count = count()
                print(f"  2D images: {count}")
            
            if hasattr(e57_file, 'data3d_count'):
                count = e57_file.data3d_count
                if callable(count):
                    count = count()
                print(f"  3D data sets: {count}")
            elif hasattr(e57_file, 'scan_count'):
                count = e57_file.scan_count
                if callable(count):
                    count = count()
                print(f"  Scans: {count}")
                        
        except Exception as e:
            print(f"  Could not read file info: {e}")
    
    def _read_scan_data(self, e57_file, scan_index: int) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Read scan data from E57 file using proper API with ignore_missing_fields."""
        try:
            if hasattr(e57_file, 'read_scan'):
                try:
                    scan_data = e57_file.read_scan(scan_index, ignore_missing_fields=True)
                    return self._extract_points_from_scan_data(scan_data)
                except Exception as e:
                    print(f"read_scan failed: {e}")
                    
            if hasattr(e57_file, 'read_scan_raw'):
                try:
                    scan_data = e57_file.read_scan_raw(scan_index, ignore_missing_fields=True)
                    return self._extract_points_from_scan_data(scan_data)
                except Exception as e:
                    print(f"read_scan_raw failed: {e}")
                    
            if hasattr(e57_file, 'data3d'):
                try:
                    for i, scan in enumerate(e57_file.data3d):
                        if i == scan_index:
                            if hasattr(e57_file, 'read_scan'):
                                scan_data = e57_file.read_scan(i, ignore_missing_fields=True)
                            else:
                                scan_data = scan
                            return self._extract_points_from_scan_data(scan_data)
                except Exception as e:
                    print(f"data3d iteration failed: {e}")
            
            return None
                
        except Exception as e:
            print(f"Error reading scan {scan_index}: {e}")
            return None
    
    def _read_all_scan_data(self, e57_file) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Try to read all scan data without specific indexing."""
        try:
            if hasattr(e57_file, 'read_scan'):
                scan_data = e57_file.read_scan(0, ignore_missing_fields=True)
                return self._extract_points_from_scan_data(scan_data)
            
            if hasattr(e57_file, 'read_scan_raw'):
                scan_data = e57_file.read_scan_raw(0, ignore_missing_fields=True)
                return self._extract_points_from_scan_data(scan_data)
                
            return None
            
        except Exception as e:
            print(f"Error in alternative reading approach: {e}")
            return None
    
    def _extract_points_from_scan_data(self, scan_data) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Extract points and colors from scan data."""
        try:
            points = None
            colors = None
            
            if isinstance(scan_data, dict):
                coord_names = [
                    ('cartesianX', 'cartesianY', 'cartesianZ'),
                    ('x', 'y', 'z'),
                    ('X', 'Y', 'Z')
                ]
                
                for x_name, y_name, z_name in coord_names:
                    if x_name in scan_data and y_name in scan_data and z_name in scan_data:
                        points = np.column_stack([
                            scan_data[x_name],
                            scan_data[y_name],
                            scan_data[z_name]
                        ])
                        break
                
                color_names = [
                    ('colorRed', 'colorGreen', 'colorBlue'),
                    ('red', 'green', 'blue'),
                    ('r', 'g', 'b')
                ]
                
                for r_name, g_name, b_name in color_names:
                    if r_name in scan_data and g_name in scan_data and b_name in scan_data:
                        colors = np.column_stack([
                            scan_data[r_name],
                            scan_data[g_name],
                            scan_data[b_name]
                        ])
                        break
                    
            elif hasattr(scan_data, 'points'):
                if hasattr(scan_data.points, 'cartesian'):
                    cart_data = scan_data.points.cartesian
                    points = np.column_stack([cart_data.x, cart_data.y, cart_data.z])
                elif hasattr(scan_data.points, 'x'):
                    points = np.column_stack([
                        scan_data.points.x,
                        scan_data.points.y,
                        scan_data.points.z
                    ])
                
                if hasattr(scan_data, 'colors') or hasattr(scan_data, 'color'):
                    color_data = getattr(scan_data, 'colors', getattr(scan_data, 'color', None))
                    if color_data is not None and hasattr(color_data, 'r'):
                        colors = np.column_stack([color_data.r, color_data.g, color_data.b])
            
            else:
                try:
                    scan_array = np.array(scan_data)
                    if len(scan_array.shape) == 2 and scan_array.shape[1] >= 3:
                        points = scan_array[:, :3]
                        if scan_array.shape[1] >= 6:
                            colors = scan_array[:, 3:6]
                except:
                    pass
            
            if points is not None and len(points) > 0:
                valid_mask = np.isfinite(points).all(axis=1)
                points = points[valid_mask]
                
                if colors is not None:
                    colors = colors[valid_mask]
                
                return points, colors
            else:
                return None
                
        except Exception as e:
            print(f"Error extracting points from scan data: {e}")
            return None
    
    def scan_and_convert_directory(self, directory: Path) -> List[Tuple[Path, Path]]:
        """
        Scan directory for E57 files and convert them - FIXED to prevent duplicates.
        """
        if not self.pye57_available:
            return []
        
        if not directory.exists():
            print(f"No E57 files found in {directory.name}")
            return []
        
        # Find E57 files - use set to prevent duplicates
        e57_files = set()
        for pattern in ['*.e57', '*.E57']:
            e57_files.update(directory.glob(pattern))
        
        e57_files = list(e57_files)  # Convert back to list
        
        if not e57_files:
            print(f"No E57 files found in {directory.name}")
            return []
        
        print(f"Found {len(e57_files)} E57 file(s) in {directory.name}")
        for e57_file in e57_files:
            print(f"Found E57 file: {e57_file.name}")
        
        conversions = []
        for e57_file in e57_files:
            # Check if already processed
            if str(e57_file.absolute()) in self.processed_files:
                print(f"Skipping already processed file: {e57_file.name}")
                converted_path = e57_file.parent / f"{e57_file.stem}_converted.ply"
                if converted_path.exists():
                    conversions.append((e57_file, converted_path))
                continue
                
            converted_path = self.convert_e57_to_ply(e57_file)
            if converted_path:
                conversions.append((e57_file, converted_path))
                print(f"Successfully converted: {e57_file.name} -> {converted_path.name}")
            else:
                print(f"Failed to convert: {e57_file.name}")
        
        return conversions
    
    def process_file_with_e57_check(self, file_path: Path) -> Path:
        """Process file with E57 conversion if needed."""
        if file_path.suffix.lower() == '.e57':
            if not self.pye57_available:
                raise Exception(f"E57 file detected but pye57 library not available: {file_path.name}")
            
            converted_path = file_path.parent / f"{file_path.stem}_converted.ply"
            if converted_path.exists():
                print(f"Using existing converted file: {converted_path.name}")
                return converted_path
            
            converted_path = self.convert_e57_to_ply(file_path)
            if converted_path is None:
                raise Exception(f"E57 conversion failed for {file_path.name}")
            
            return converted_path
        else:
            return file_path
    
    def get_conversion_info(self) -> Dict[str, Any]:
        """Get information about conversions performed."""
        return {
            'pye57_available': self.pye57_available,
            'conversions_performed': len(self.conversion_log),
            'conversion_log': self.conversion_log
        }
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary conversion files if needed."""
        pass


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING E57 CONVERTER - FIXED VERSION")
    print("=" * 60)
    
    converter = E57Converter()
    
    if converter.pye57_available:
        print("E57 converter ready")
        info = converter.get_conversion_info()
        print(f"pye57 available: {info['pye57_available']}")
        print(f"Conversions performed: {info['conversions_performed']}")
    else:
        print("E57 support not available - install pye57 library")
    
    print("E57 converter test completed")