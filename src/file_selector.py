"""
Intelligent File Selector for Mill Analysis Project
Automatically discovers and assigns files based on directory contents
Provides user interaction for file selection when multiple options exist
Supports E57, PLY, PCD, and XYZ formats
Compatible with single-scan and dual-scan workflows
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from config import get_file_paths, FileConfig


class IntelligentFileSelector:
    """Intelligent file selector with automatic discovery and user interaction."""
    
    def __init__(self, dual_scan_mode: bool = False):
        """
        Initialize file selector.
        
        Args:
            dual_scan_mode: If True, selects two inner scan files. If False, selects one (default for compatibility)
        """
        self.file_config = FileConfig()
        self.file_paths = get_file_paths()
        self.supported_formats = self.file_config.SUPPORTED_FORMATS
        self.dual_scan_mode = dual_scan_mode
        
        # File assignment results
        self.selected_reference = None
        self.selected_inner_scan = None  # Primary inner scan (always used)
        self.selected_inner_scan2 = None  # Secondary inner scan (only for dual-scan mode)
        self.selection_summary = {}
    
    def discover_files_in_directory(self, directory: Path, 
                                   description: str = "files") -> List[Path]:
        """
        Discover all supported point cloud files in a directory.
        
        Args:
            directory: Directory to scan
            description: Description for logging
            
        Returns:
            List of discovered file paths, sorted by modification time (newest first)
        """
        discovered_files = []
        
        if not directory.exists():
            print(f"WARNING: {description} directory not found: {directory}")
            return discovered_files
        
        print(f"\nScanning {description} directory: {directory.name}")
        
        try:
            # Find all supported files
            for file_path in directory.iterdir():
                if (file_path.is_file() and 
                    file_path.suffix.lower() in self.supported_formats):
                    discovered_files.append(file_path)
            
            # Sort by modification time (newest first) for consistent ordering
            discovered_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            if discovered_files:
                print(f"Found {len(discovered_files)} supported file(s):")
                for i, file_path in enumerate(discovered_files, 1):
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    print(f"  {i}. {file_path.name} ({size_mb:.1f} MB, {mod_time.strftime('%Y-%m-%d %H:%M')})")
            else:
                print(f"No supported files found in {description} directory")
                print(f"Supported formats: {self.supported_formats}")
        
        except Exception as e:
            print(f"ERROR scanning {description} directory: {e}")
        
        return discovered_files
    
    def prompt_user_selection(self, files: List[Path], 
                            selection_type: str,
                            max_selections: int = 1) -> List[Path]:
        """
        Prompt user to select file(s) from a list.
        
        Args:
            files: List of available files
            selection_type: Description of what's being selected
            max_selections: Maximum number of files to select
            
        Returns:
            List of selected files
        """
        if not files:
            print(f"No files available for {selection_type} selection")
            return []
        
        if len(files) == 1 and max_selections >= 1:
            print(f"Only one file available for {selection_type}, auto-selecting: {files[0].name}")
            return files
        
        print(f"\n=== {selection_type.upper()} SELECTION ===")
        print(f"Please select {'file(s)' if max_selections > 1 else 'a file'} for {selection_type}:")
        
        # Display options
        for i, file_path in enumerate(files, 1):
            size_mb = file_path.stat().st_size / (1024 * 1024)
            mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            print(f"  {i}. {file_path.name}")
            print(f"     Size: {size_mb:.1f} MB, Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"     Format: {file_path.suffix.upper()}")
        
        selected_files = []
        
        try:
            if max_selections == 1:
                # Single selection
                while True:
                    try:
                        choice = input(f"\nEnter choice (1-{len(files)}): ").strip()
                        choice_num = int(choice)
                        if 1 <= choice_num <= len(files):
                            selected_files = [files[choice_num - 1]]
                            print(f"Selected: {selected_files[0].name}")
                            break
                        else:
                            print(f"Please enter a number between 1 and {len(files)}")
                    except ValueError:
                        print("Please enter a valid number")
                    except KeyboardInterrupt:
                        print("\nSelection cancelled by user")
                        return []
            else:
                # Multiple selection
                print(f"\nYou can select up to {max_selections} files.")
                print("Enter numbers separated by spaces (e.g., '1 3' for files 1 and 3)")
                print("Or enter a single number to select just one file")
                
                while True:
                    try:
                        choices_input = input(f"\nEnter choice(s) (1-{len(files)}): ").strip()
                        if not choices_input:
                            continue
                        
                        choice_nums = [int(x.strip()) for x in choices_input.split()]
                        
                        # Validate choices
                        if all(1 <= num <= len(files) for num in choice_nums):
                            if len(choice_nums) <= max_selections:
                                # Remove duplicates and maintain order
                                unique_choices = []
                                for num in choice_nums:
                                    if num not in unique_choices:
                                        unique_choices.append(num)
                                
                                selected_files = [files[num - 1] for num in unique_choices]
                                print(f"Selected {len(selected_files)} file(s):")
                                for file_path in selected_files:
                                    print(f"  - {file_path.name}")
                                break
                            else:
                                print(f"Please select at most {max_selections} files")
                        else:
                            print(f"Please enter numbers between 1 and {len(files)}")
                    except ValueError:
                        print("Please enter valid numbers separated by spaces")
                    except KeyboardInterrupt:
                        print("\nSelection cancelled by user")
                        return []
        
        except Exception as e:
            print(f"ERROR during file selection: {e}")
            return []
        
        return selected_files
    
    def select_reference_file(self) -> Optional[Path]:
        """
        Select reference file intelligently.
        
        Returns:
            Selected reference file path or None if failed
        """
        print("\n" + "="*60)
        print("INTELLIGENT REFERENCE FILE SELECTION")
        print("="*60)
        
        # Discover reference files
        ref_dir = self.file_paths['reference_dir']
        discovered_files = self.discover_files_in_directory(ref_dir, "reference")
        
        if not discovered_files:
            # Fallback: check if specific reference file exists
            specific_ref = self.file_paths.get('reference')
            if specific_ref and specific_ref.exists():
                print(f"Using configured reference file: {specific_ref.name}")
                self.selected_reference = specific_ref
                return specific_ref
            else:
                print("ERROR: No reference files found")
                return None
        
        elif len(discovered_files) == 1:
            # Auto-select single file
            self.selected_reference = discovered_files[0]
            print(f"Auto-selected reference file: {self.selected_reference.name}")
            return self.selected_reference
        
        else:
            # Multiple files - ask user
            selected_files = self.prompt_user_selection(
                discovered_files, 
                "reference file", 
                max_selections=1
            )
            
            if selected_files:
                self.selected_reference = selected_files[0]
                print(f"User selected reference file: {self.selected_reference.name}")
                return self.selected_reference
            else:
                print("ERROR: No reference file selected")
                return None
    
    def select_inner_scan_files(self) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Select inner scan files intelligently.
        
        For single-scan mode (default, compatible):
        - Returns (selected_file, None)
        
        For dual-scan mode:
        - 1 file: Assign to both inner_scan and inner_scan2
        - 2+ files: Ask user to select 1 or 2 files
        
        Returns:
            Tuple of (inner_scan_path, inner_scan2_path)
        """
        print("\n" + "="*60)
        if self.dual_scan_mode:
            print("INTELLIGENT DUAL INNER SCAN FILE SELECTION")
        else:
            print("INTELLIGENT INNER SCAN FILE SELECTION (SINGLE SCAN MODE)")
        print("="*60)
        
        # Discover inner scan files
        scans_dir = self.file_paths['inner_scans_dir']
        discovered_files = self.discover_files_in_directory(scans_dir, "inner scan")
        
        if not discovered_files:
            # Fallback: check if specific inner scan file exists
            specific_scan = self.file_paths.get('inner_scan')
            if specific_scan and specific_scan.exists():
                print(f"Using configured inner scan file: {specific_scan.name}")
                self.selected_inner_scan = specific_scan
                if self.dual_scan_mode:
                    self.selected_inner_scan2 = specific_scan
                    return specific_scan, specific_scan
                else:
                    return specific_scan, None
            else:
                print("ERROR: No inner scan files found")
                return None, None
        
        elif len(discovered_files) == 1:
            # Single file
            self.selected_inner_scan = discovered_files[0]
            print(f"Single file found: {self.selected_inner_scan.name}")
            
            if self.dual_scan_mode:
                self.selected_inner_scan2 = discovered_files[0]
                print("Note: Same file will be used for both inner scans (dual-scan mode)")
                return self.selected_inner_scan, self.selected_inner_scan2
            else:
                print("Using for primary inner scan (single-scan mode)")
                return self.selected_inner_scan, None
        
        elif not self.dual_scan_mode:
            # Single-scan mode with multiple files - select one
            print(f"Found {len(discovered_files)} inner scan files")
            print("Please select 1 file for inner scan processing:")
            
            selected_files = self.prompt_user_selection(
                discovered_files, 
                "inner scan file", 
                max_selections=1
            )
            
            if selected_files:
                self.selected_inner_scan = selected_files[0]
                print(f"Selected inner scan file: {self.selected_inner_scan.name}")
                return self.selected_inner_scan, None
            else:
                print("ERROR: No inner scan file selected")
                return None, None
        
        else:
            # Dual-scan mode with multiple files
            if len(discovered_files) == 2:
                # Perfect - two files, auto-assign
                self.selected_inner_scan = discovered_files[0]  # Newest file
                self.selected_inner_scan2 = discovered_files[1]  # Second newest file
                print(f"Two files found - auto-assigning:")
                print(f"  inner_scan (primary): {self.selected_inner_scan.name} (newer)")
                print(f"  inner_scan2 (secondary): {self.selected_inner_scan2.name} (older)")
                return self.selected_inner_scan, self.selected_inner_scan2
            else:
                # Multiple files - ask user to select 1 or 2
                print(f"Found {len(discovered_files)} inner scan files")
                print("Please select 1 or 2 files for dual-scan processing")
                print("(Select 1 file to use for both scans, or 2 different files)")
                
                selected_files = self.prompt_user_selection(
                    discovered_files, 
                    "inner scan files", 
                    max_selections=2
                )
                
                if len(selected_files) == 2:
                    self.selected_inner_scan = selected_files[0]
                    self.selected_inner_scan2 = selected_files[1]
                    print(f"User selected:")
                    print(f"  inner_scan (primary): {self.selected_inner_scan.name}")
                    print(f"  inner_scan2 (secondary): {self.selected_inner_scan2.name}")
                    return self.selected_inner_scan, self.selected_inner_scan2
                
                elif len(selected_files) == 1:
                    self.selected_inner_scan = selected_files[0]
                    self.selected_inner_scan2 = selected_files[0]
                    print(f"User selected single file for both scans: {self.selected_inner_scan.name}")
                    return self.selected_inner_scan, self.selected_inner_scan2
                
                else:
                    print("ERROR: No inner scan files selected")
                    return None, None
    
    def perform_intelligent_file_selection(self) -> Dict[str, any]:
        """
        Perform complete intelligent file selection.
        
        Returns:
            Dictionary with selected file paths and metadata
        """
        print("\n" + "="*80)
        print("MILL ANALYSIS - INTELLIGENT FILE SELECTION SYSTEM")
        if self.dual_scan_mode:
            print("Running in DUAL-SCAN MODE")
        else:
            print("Running in SINGLE-SCAN MODE (Compatible with current workflow)")
        print("="*80)
        print("Automatically discovering and assigning files based on directory contents...")
        
        # Select reference file
        reference_file = self.select_reference_file()
        
        # Select inner scan files
        inner_scan, inner_scan2 = self.select_inner_scan_files()
        
        # Compile results
        success = reference_file is not None and inner_scan is not None
        
        if self.dual_scan_mode:
            success = success and inner_scan2 is not None
        
        if success:
            self.selection_summary = {
                'reference': reference_file,
                'inner_scan': inner_scan,
                'inner_scan2': inner_scan2,  # None for single-scan mode
                'selection_successful': True,
                'dual_scan_mode': self.dual_scan_mode,
                'same_inner_scan': inner_scan == inner_scan2 if inner_scan2 else False
            }
            
            # Print summary
            print("\n" + "="*60)
            print("FILE SELECTION SUMMARY")
            print("="*60)
            print(f"Reference file: {reference_file.name}")
            print(f"Inner scan (primary): {inner_scan.name}")
            
            if self.dual_scan_mode:
                print(f"Inner scan 2 (secondary): {inner_scan2.name if inner_scan2 else 'None'}")
                if inner_scan == inner_scan2:
                    print("Note: Same file used for both inner scans")
                else:
                    print("Note: Different files used for inner scans")
            else:
                print("Mode: Single-scan (compatible with current workflow)")
            
            print("="*60)
            print("File selection completed successfully!")
            
        else:
            self.selection_summary = {
                'reference': reference_file,
                'inner_scan': inner_scan,
                'inner_scan2': inner_scan2,
                'selection_successful': False,
                'dual_scan_mode': self.dual_scan_mode,
                'error': 'Failed to select required files'
            }
            
            print("\n" + "="*60)
            print("FILE SELECTION FAILED")
            print("="*60)
            print("Could not select all required files")
            if not reference_file:
                print("- Reference file: MISSING")
            if not inner_scan:
                print("- Inner scan (primary): MISSING")
            if self.dual_scan_mode and not inner_scan2:
                print("- Inner scan 2 (secondary): MISSING")
        
        return self.selection_summary
    
    def get_selected_files(self) -> Dict[str, Optional[Path]]:
        """
        Get the currently selected files.
        
        Returns:
            Dictionary with selected file paths
        """
        result = {
            'reference': self.selected_reference,
            'inner_scan': self.selected_inner_scan,
        }
        
        if self.dual_scan_mode:
            result['inner_scan2'] = self.selected_inner_scan2
        
        return result
    
    def has_valid_selection(self) -> bool:
        """
        Check if a valid file selection has been made.
        
        Returns:
            True if all required files are selected
        """
        basic_valid = (self.selected_reference is not None and 
                      self.selected_inner_scan is not None)
        
        if self.dual_scan_mode:
            return basic_valid and self.selected_inner_scan2 is not None
        else:
            return basic_valid


# Convenience functions for easy integration
def perform_intelligent_file_selection(dual_scan_mode: bool = False) -> Dict[str, any]:
    """
    Convenience function to perform intelligent file selection.
    
    Args:
        dual_scan_mode: If True, selects two inner scan files. If False, selects one (default)
    
    Returns:
        Dictionary with selected file paths and metadata
    """
    selector = IntelligentFileSelector(dual_scan_mode=dual_scan_mode)
    return selector.perform_intelligent_file_selection()


def get_single_scan_files() -> Dict[str, Optional[Path]]:
    """
    Convenience function for single-scan file selection (compatible with current workflow).
    
    Returns:
        Dictionary with selected reference and inner_scan paths
    """
    selector = IntelligentFileSelector(dual_scan_mode=False)
    results = selector.perform_intelligent_file_selection()
    
    if results.get('selection_successful', False):
        return {
            'reference': results['reference'],
            'inner_scan': results['inner_scan']
        }
    else:
        return {'reference': None, 'inner_scan': None}


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING INTELLIGENT FILE SELECTOR")
    print("=" * 60)
    
    # Test single-scan mode (default, compatible)
    print("\n--- Testing Single-Scan Mode (Compatible) ---")
    selector = IntelligentFileSelector(dual_scan_mode=False)
    
    print(f"Supported formats: {selector.supported_formats}")
    print(f"Reference directory: {selector.file_paths['reference_dir']}")
    print(f"Inner scans directory: {selector.file_paths['inner_scans_dir']}")
    print(f"Dual-scan mode: {selector.dual_scan_mode}")
    
    # Perform selection
    results = selector.perform_intelligent_file_selection()
    
    if results.get('selection_successful', False):
        print("\nSuccess: Intelligent file selection completed!")
        selected_files = selector.get_selected_files()
        for key, path in selected_files.items():
            if path:
                print(f"  {key}: {path.name}")
    else:
        print("\nError: File selection failed")
    
    print("\nIntelligent File Selector - Ready for pipeline integration")
    print("Compatible with existing single-scan workflow")
