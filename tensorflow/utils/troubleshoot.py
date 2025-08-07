#!/usr/bin/env python3
"""
Troubleshooting utilities for TensorFlow DewarpNet setup issues.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


class TroubleshootingGuide:
    """Interactive troubleshooting guide for common setup issues."""
    
    def __init__(self):
        self.solutions = {
            'conda_not_found': self._fix_conda_not_found,
            'environment_missing': self._fix_environment_missing,
            'tensorflow_import_error': self._fix_tensorflow_import,
            'gpu_not_detected': self._fix_gpu_not_detected,
            'cuda_version_mismatch': self._fix_cuda_version_mismatch,
            'memory_error': self._fix_memory_error,
            'dataset_not_found': self._fix_dataset_not_found,
            'permission_error': self._fix_permission_error
        }
    
    def diagnose_issue(self) -> str:
        """Interactive diagnosis of common issues."""
        print("=== TensorFlow DewarpNet Troubleshooting ===\n")
        print("Let's diagnose your issue. Please select the problem you're experiencing:\n")
        
        issues = [
            ("conda_not_found", "conda command not found"),
            ("environment_missing", "Conda environment doesn't exist or can't be activated"),
            ("tensorflow_import_error", "Cannot import TensorFlow or getting import errors"),
            ("gpu_not_detected", "GPU not detected by TensorFlow"),
            ("cuda_version_mismatch", "CUDA version compatibility issues"),
            ("memory_error", "Out of memory errors during training"),
            ("dataset_not_found", "Dataset files not found or inaccessible"),
            ("permission_error", "Permission denied errors")
        ]
        
        for i, (key, description) in enumerate(issues, 1):
            print(f"{i}. {description}")
        
        try:
            choice = input("\nEnter the number of your issue (1-8): ").strip()
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(issues):
                issue_key = issues[choice_idx][0]
                return self.solutions[issue_key]()
            else:
                print("Invalid choice. Please run the script again.")
                return ""
                
        except (ValueError, KeyboardInterrupt):
            print("\nTroubleshooting cancelled.")
            return ""
    
    def _fix_conda_not_found(self) -> str:
        """Fix conda not found issues."""
        print("\n=== Fixing Conda Installation Issues ===\n")
        
        # Check if conda is installed but not in PATH
        possible_conda_paths = [
            Path.home() / "miniconda3" / "bin" / "conda",
            Path.home() / "anaconda3" / "bin" / "conda",
            Path("/opt/miniconda3/bin/conda"),
            Path("/opt/anaconda3/bin/conda")
        ]
        
        found_conda = None
        for conda_path in possible_conda_paths:
            if conda_path.exists():
                found_conda = conda_path
                break
        
        if found_conda:
            print(f"✓ Found conda at: {found_conda}")
            print("\nTo fix PATH issue, add this to your ~/.bashrc or ~/.zshrc:")
            print(f'export PATH="{found_conda.parent}:$PATH"')
            print("\nThen restart your terminal or run: source ~/.bashrc")
            return "conda_path_fix"
        else:
            print("❌ Conda not found. Installing Miniconda...")
            return self._install_miniconda()
    
    def _install_miniconda(self) -> str:
        """Guide user through Miniconda installation."""
        system = platform.system().lower()
        arch = platform.machine().lower()
        
        print(f"\nDetected system: {system} {arch}")
        
        if system == "linux":
            if "x86_64" in arch or "amd64" in arch:
                installer_url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
            elif "aarch64" in arch or "arm64" in arch:
                installer_url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
            else:
                print(f"Unsupported architecture: {arch}")
                return "unsupported_arch"
        elif system == "darwin":  # macOS
            if "arm64" in arch:
                installer_url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
            else:
                installer_url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
        elif system == "windows":
            installer_url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
        else:
            print(f"Unsupported operating system: {system}")
            return "unsupported_os"
        
        print(f"\nTo install Miniconda:")
        print(f"1. Download: {installer_url}")
        
        if system in ["linux", "darwin"]:
            print("2. Run: bash Miniconda3-latest-*.sh")
            print("3. Follow the installation prompts")
            print("4. Restart your terminal")
            print("5. Run: conda --version")
        else:
            print("2. Run the downloaded .exe file")
            print("3. Follow the installation wizard")
            print("4. Restart your command prompt")
            print("5. Run: conda --version")
        
        return "miniconda_install_guide"
    
    def _fix_environment_missing(self) -> str:
        """Fix missing conda environment."""
        print("\n=== Fixing Missing Conda Environment ===\n")
        
        env_name = "dewarpnet_tf_kiro_full"
        
        print(f"Creating conda environment: {env_name}")
        print("\nRun these commands:")
        print(f"1. conda create -n {env_name} python=3.9 -y")
        print(f"2. conda activate {env_name}")
        print("3. pip install -r tensorflow/requirements_tf.txt")
        print("4. conda install cudatoolkit cudnn -c conda-forge -y")
        
        # Offer to run automatically
        try:
            auto_create = input("\nWould you like me to create the environment automatically? (y/n): ").strip().lower()
            if auto_create == 'y':
                return self._auto_create_environment(env_name)
        except KeyboardInterrupt:
            pass
        
        return "manual_env_creation"
    
    def _auto_create_environment(self, env_name: str) -> str:
        """Automatically create conda environment."""
        print(f"\nCreating environment {env_name}...")
        
        try:
            # Create environment
            subprocess.run(f"conda create -n {env_name} python=3.9 -y", 
                         shell=True, check=True)
            print("✓ Environment created")
            
            # Install requirements
            requirements_path = Path("tensorflow/requirements_tf.txt")
            if requirements_path.exists():
                subprocess.run(f"conda run -n {env_name} pip install -r {requirements_path}", 
                             shell=True, check=True)
                print("✓ Python packages installed")
            
            # Install CUDA packages
            subprocess.run(f"conda install -n {env_name} cudatoolkit cudnn -c conda-forge -y", 
                         shell=True, check=True)
            print("✓ CUDA packages installed")
            
            print(f"\n✓ Environment {env_name} created successfully!")
            print(f"To activate: conda activate {env_name}")
            
            return "environment_created"
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error creating environment: {e}")
            return "environment_creation_failed"
    
    def _fix_tensorflow_import(self) -> str:
        """Fix TensorFlow import issues."""
        print("\n=== Fixing TensorFlow Import Issues ===\n")
        
        # Check if environment is activated
        current_env = os.environ.get('CONDA_DEFAULT_ENV', 'base')
        print(f"Current environment: {current_env}")
        
        if current_env == 'base':
            print("⚠️  You're in the base environment. Activate the correct environment:")
            print("conda activate dewarpnet_tf_kiro_full")
            return "wrong_environment"
        
        # Try to import TensorFlow and diagnose
        try:
            import tensorflow as tf
            print(f"✓ TensorFlow {tf.__version__} imported successfully")
            return "tensorflow_working"
        except ImportError as e:
            print(f"❌ TensorFlow import error: {e}")
            
            # Common fixes
            print("\nTrying common fixes:")
            print("1. Reinstall TensorFlow:")
            print("   pip uninstall tensorflow")
            print("   pip install tensorflow>=2.10.0")
            print("\n2. Check for conflicting packages:")
            print("   pip list | grep tensor")
            print("\n3. Clear pip cache:")
            print("   pip cache purge")
            
            return "tensorflow_import_failed"
    
    def _fix_gpu_not_detected(self) -> str:
        """Fix GPU detection issues."""
        print("\n=== Fixing GPU Detection Issues ===\n")
        
        # Check NVIDIA driver
        try:
            result = subprocess.run("nvidia-smi", shell=True, 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✓ NVIDIA driver is working")
                print(result.stdout.split('\n')[0])  # First line with driver info
            else:
                print("❌ nvidia-smi failed - driver issue")
                return self._fix_nvidia_driver()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ nvidia-smi not found - driver not installed")
            return self._fix_nvidia_driver()
        
        # Check CUDA installation
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"✓ TensorFlow detected {len(gpus)} GPU(s)")
                for i, gpu in enumerate(gpus):
                    print(f"  GPU {i}: {gpu.name}")
                return "gpu_working"
            else:
                print("❌ TensorFlow cannot see GPU")
                return self._fix_tensorflow_gpu()
        except ImportError:
            print("❌ Cannot import TensorFlow")
            return "tensorflow_not_installed"
    
    def _fix_nvidia_driver(self) -> str:
        """Fix NVIDIA driver issues."""
        print("\n=== Fixing NVIDIA Driver Issues ===\n")
        
        system = platform.system().lower()
        
        if system == "linux":
            print("For Ubuntu/Debian:")
            print("1. sudo apt update")
            print("2. sudo apt install nvidia-driver-470")  # or latest
            print("3. sudo reboot")
            print("\nFor other distributions, check your package manager.")
        elif system == "windows":
            print("1. Visit: https://www.nvidia.com/drivers")
            print("2. Download and install the latest driver")
            print("3. Restart your computer")
        else:
            print("Please install NVIDIA drivers for your operating system")
        
        return "nvidia_driver_fix"
    
    def _fix_tensorflow_gpu(self) -> str:
        """Fix TensorFlow GPU support."""
        print("\n=== Fixing TensorFlow GPU Support ===\n")
        
        print("Installing CUDA support packages:")
        print("1. conda install cudatoolkit cudnn -c conda-forge")
        print("2. pip install tensorflow[and-cuda]")
        print("\nOr try:")
        print("1. pip uninstall tensorflow")
        print("2. pip install tensorflow-gpu>=2.10.0")
        
        return "tensorflow_gpu_fix"
    
    def _fix_cuda_version_mismatch(self) -> str:
        """Fix CUDA version compatibility issues."""
        print("\n=== Fixing CUDA Version Issues ===\n")
        
        # Check system CUDA version
        try:
            result = subprocess.run("nvcc --version", shell=True,
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("System CUDA version:")
                print(result.stdout)
            else:
                print("❌ nvcc not found - CUDA toolkit not installed")
        except:
            print("❌ Could not check CUDA version")
        
        # Check TensorFlow CUDA requirements
        print("\nTensorFlow CUDA compatibility:")
        print("- TensorFlow 2.10: CUDA 11.2, cuDNN 8.1")
        print("- TensorFlow 2.11: CUDA 11.2, cuDNN 8.1")
        print("- TensorFlow 2.12: CUDA 11.8, cuDNN 8.6")
        
        print("\nRecommended fix:")
        print("1. Use conda to manage CUDA versions:")
        print("   conda install cudatoolkit=11.8 cudnn=8.6 -c conda-forge")
        print("2. Or use tensorflow[and-cuda] which includes compatible versions")
        
        return "cuda_version_fix"
    
    def _fix_memory_error(self) -> str:
        """Fix memory-related errors."""
        print("\n=== Fixing Memory Issues ===\n")
        
        print("Common memory fixes:")
        print("1. Reduce batch size in training scripts")
        print("2. Enable GPU memory growth:")
        print("   tf.config.experimental.set_memory_growth(gpu, True)")
        print("3. Use mixed precision training:")
        print("   tf.keras.mixed_precision.set_global_policy('mixed_float16')")
        print("4. Clear GPU memory between runs:")
        print("   tf.keras.backend.clear_session()")
        
        # Check available memory
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"\nGPU memory configuration:")
                for gpu in gpus:
                    print(f"  {gpu.name}")
        except:
            pass
        
        return "memory_fix"
    
    def _fix_dataset_not_found(self) -> str:
        """Fix dataset access issues."""
        print("\n=== Fixing Dataset Issues ===\n")
        
        dataset_path = "/home/argus/Workspace/dataset"
        print(f"Expected dataset location: {dataset_path}")
        
        if not Path(dataset_path).exists():
            print(f"❌ Dataset directory not found: {dataset_path}")
            print("\nOptions:")
            print("1. Create symbolic link to your dataset:")
            print(f"   ln -s /path/to/your/dataset {dataset_path}")
            print("2. Update dataset path in scripts")
            print("3. Download dataset from official source")
        else:
            print("✓ Dataset directory exists")
            
            # Check subdirectories
            expected_dirs = ["doc3d", "doc3d_100", "doc3d_1000", "input_crop"]
            for dir_name in expected_dirs:
                dir_path = Path(dataset_path) / dir_name
                if dir_path.exists():
                    print(f"✓ {dir_name}")
                else:
                    print(f"❌ {dir_name} not found")
        
        return "dataset_fix"
    
    def _fix_permission_error(self) -> str:
        """Fix permission-related errors."""
        print("\n=== Fixing Permission Issues ===\n")
        
        print("Common permission fixes:")
        print("1. Check file/directory permissions:")
        print("   ls -la /path/to/problematic/file")
        print("2. Fix ownership:")
        print("   sudo chown -R $USER:$USER /path/to/directory")
        print("3. Fix permissions:")
        print("   chmod -R 755 /path/to/directory")
        print("4. For conda environments:")
        print("   sudo chown -R $USER:$USER ~/miniconda3")
        
        return "permission_fix"
    
    def run_automated_fixes(self) -> Dict[str, str]:
        """Run automated fixes for common issues."""
        print("=== Running Automated Fixes ===\n")
        
        fixes_applied = {}
        
        # Fix 1: Update pip and setuptools
        try:
            print("Updating pip and setuptools...")
            subprocess.run("pip install --upgrade pip setuptools", shell=True, check=True)
            fixes_applied['pip_update'] = 'success'
            print("✓ pip and setuptools updated")
        except:
            fixes_applied['pip_update'] = 'failed'
            print("❌ Failed to update pip")
        
        # Fix 2: Clear pip cache
        try:
            print("Clearing pip cache...")
            subprocess.run("pip cache purge", shell=True, check=True)
            fixes_applied['pip_cache'] = 'success'
            print("✓ pip cache cleared")
        except:
            fixes_applied['pip_cache'] = 'failed'
            print("❌ Failed to clear pip cache")
        
        # Fix 3: Reinstall TensorFlow
        try:
            print("Reinstalling TensorFlow...")
            subprocess.run("pip uninstall tensorflow -y", shell=True)
            subprocess.run("pip install tensorflow>=2.10.0", shell=True, check=True)
            fixes_applied['tensorflow_reinstall'] = 'success'
            print("✓ TensorFlow reinstalled")
        except:
            fixes_applied['tensorflow_reinstall'] = 'failed'
            print("❌ Failed to reinstall TensorFlow")
        
        return fixes_applied


def main():
    """Main function for interactive troubleshooting."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TensorFlow DewarpNet Troubleshooting")
    parser.add_argument("--auto-fix", action="store_true",
                       help="Run automated fixes")
    parser.add_argument("--issue", choices=['conda', 'env', 'tensorflow', 'gpu', 'cuda', 'memory', 'dataset', 'permission'],
                       help="Directly address a specific issue")
    
    args = parser.parse_args()
    
    guide = TroubleshootingGuide()
    
    if args.auto_fix:
        results = guide.run_automated_fixes()
        print(f"\nAutomated fixes completed: {results}")
    elif args.issue:
        issue_map = {
            'conda': 'conda_not_found',
            'env': 'environment_missing',
            'tensorflow': 'tensorflow_import_error',
            'gpu': 'gpu_not_detected',
            'cuda': 'cuda_version_mismatch',
            'memory': 'memory_error',
            'dataset': 'dataset_not_found',
            'permission': 'permission_error'
        }
        result = guide.solutions[issue_map[args.issue]]()
        print(f"\nTroubleshooting result: {result}")
    else:
        guide.diagnose_issue()


if __name__ == "__main__":
    main()