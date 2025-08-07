#!/usr/bin/env python3
"""
Environment setup script for TensorFlow DewarpNet implementation.
Creates conda environment and installs dependencies.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, check=True, shell=True):
    """Run shell command with error handling."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=shell, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Error output: {e.stderr}")
        if check:
            sys.exit(1)
        return e


def check_conda():
    """Check if conda is available."""
    try:
        result = subprocess.run("conda --version", shell=True, 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Found conda: {result.stdout.strip()}")
            return True
    except:
        pass
    
    print("Error: conda not found. Please install Anaconda or Miniconda first.")
    print("Visit: https://docs.conda.io/en/latest/miniconda.html")
    return False


def create_conda_env(env_name="dewarpnet_tf", python_version="3.9"):
    """Create conda environment with specified Python version."""
    print(f"Creating conda environment: {env_name}")
    
    # Check if environment already exists
    result = run_command(f"conda env list | grep {env_name}", check=False)
    if result.returncode == 0:
        print(f"Environment {env_name} already exists.")
        response = input("Do you want to remove and recreate it? (y/n): ")
        if response.lower() == 'y':
            run_command(f"conda env remove -n {env_name}")
        else:
            return env_name
    
    # Create new environment
    run_command(f"conda create -n {env_name} python={python_version} -y")
    return env_name


def install_dependencies(env_name, requirements_file):
    """Install dependencies in conda environment."""
    print(f"Installing dependencies from {requirements_file}")
    
    # Install pip packages
    run_command(f"conda run -n {env_name} pip install -r {requirements_file}")
    
    # Install additional conda packages for better GPU support
    run_command(f"conda install -n {env_name} cudatoolkit cudnn -c conda-forge -y")


def setup_dataset_links(dataset_path="/home/argus/Workspace/dataset"):
    """Create symbolic links to dataset directories."""
    print("Setting up dataset symbolic links...")
    
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"Warning: Dataset path {dataset_path} does not exist.")
        print("Please ensure the dataset is available at the specified location.")
        return
    
    # Create tensorflow/data directory
    data_dir = Path("tensorflow/data")
    data_dir.mkdir(exist_ok=True)
    
    # Create symbolic links for different dataset sizes
    links_to_create = [
        ("doc3d", "doc3d"),
        ("doc3d_100", "doc3d_100"), 
        ("doc3d_1000", "doc3d_1000"),
        ("input_crop", "input_crop")
    ]
    
    for link_name, target_name in links_to_create:
        link_path = data_dir / link_name
        target_path = dataset_path / target_name
        
        if target_path.exists():
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            
            # Create symbolic link
            if os.name == 'nt':  # Windows
                run_command(f'mklink /D "{link_path}" "{target_path}"', check=False)
            else:  # Unix-like systems
                link_path.symlink_to(target_path)
            
            print(f"Created link: {link_path} -> {target_path}")
        else:
            print(f"Warning: Target {target_path} does not exist, skipping link creation")


def main():
    parser = argparse.ArgumentParser(description="Setup TensorFlow DewarpNet environment")
    parser.add_argument("--env-name", default="dewarpnet_tf", 
                       help="Conda environment name")
    parser.add_argument("--python-version", default="3.9",
                       help="Python version for conda environment")
    parser.add_argument("--dataset-path", default="/home/argus/Workspace/dataset",
                       help="Path to dataset directory")
    parser.add_argument("--skip-conda", action="store_true",
                       help="Skip conda environment creation")
    
    args = parser.parse_args()
    
    print("=== TensorFlow DewarpNet Environment Setup ===")
    
    # Get current directory
    current_dir = Path.cwd()
    requirements_file = current_dir / "tensorflow" / "requirements_tf.txt"
    
    if not requirements_file.exists():
        print(f"Error: Requirements file not found at {requirements_file}")
        sys.exit(1)
    
    if not args.skip_conda:
        # Check conda availability
        if not check_conda():
            sys.exit(1)
        
        # Create conda environment
        env_name = create_conda_env(args.env_name, args.python_version)
        
        # Install dependencies
        install_dependencies(env_name, requirements_file)
        
        print(f"\nEnvironment setup complete!")
        print(f"To activate the environment, run:")
        print(f"conda activate {env_name}")
    
    # Setup dataset links
    setup_dataset_links(args.dataset_path)
    
    print("\n=== Setup Complete ===")
    print("Next steps:")
    print("1. Activate the conda environment: conda activate dewarpnet_tf")
    print("2. Run GPU validation: python tensorflow/utils/gpu_utils.py")
    print("3. Test data loading: python tensorflow/utils/test_setup.py")


if __name__ == "__main__":
    main()