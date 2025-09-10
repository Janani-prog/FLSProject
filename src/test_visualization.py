"""
Test visualization setup and matplotlib backend
"""

import matplotlib
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"Matplotlib backend: {matplotlib.get_backend()}")

# Set backend explicitly
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use non-interactive backend for file saving
print(f"Backend after setting: {matplotlib.get_backend()}")

# Test basic plot creation
import numpy as np

try:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y)
    ax.set_title('Test Plot')
    
    # Save test plot
    output_path = 'test_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"SUCCESS: Test plot saved to {output_path}")
    
except Exception as e:
    print(f"ERROR: Failed to create test plot: {str(e)}")

# Test if output directories exist
from pathlib import Path
from config import get_file_paths

try:
    file_paths = get_file_paths()
    viz_dir = file_paths['visualizations_dir']
    
    print(f"Visualization directory: {viz_dir}")
    print(f"Directory exists: {viz_dir.exists()}")
    
    if not viz_dir.exists():
        viz_dir.mkdir(parents=True, exist_ok=True)
        print("Created visualization directory")
        
except Exception as e:
    print(f"ERROR: Problem with directories: {str(e)}")
