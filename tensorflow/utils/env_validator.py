#!/usr/bin/env python3
"""
Comprehensive environment validation and troubleshooting utilities.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


class EnvironmentValidator:
    """Comprehensive environment validation for TensorFlow DewarpNet."""
    
    def __init__(self):
        self.results = {}
        self.errors = []
        self.warnings = []
        
    def check_system_requirements(self) -> bool:
        """Check basic system requirements."""
        print("Checking system requirements...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            self.errors.append(f"Python {python_version.major}.{python_version.minor} is too old. Requires Python 3.8+")
            return False
        
        print(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check operating system
        os_name = platform.system()
        print(f"✓ Operating System: {os_name} {platform.release()}")
        
        # Check available memory
        try:
            if os_name == "Linux":
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                    for line in meminfo.split('\n'):
                        if 'MemTotal:' in line:
                            mem_kb = int(line.split()[1])
                            mem_gb = mem_kb / (1024 * 1024)
                            print(f"✓ System Memory: {mem_gb:.1f} GB")
                            if mem_gb < 8:
                                self.warnings.append(f"Low system memory ({mem_gb:.1f} GB). Recommend 16GB+ for training")
                            break
        except:
            print("⚠️  Could not determine system memory")
        
        return True
    
    def check_conda_installation(self) -> bool:
        """Check conda installation and configuration."""
        print("Checking conda installation...")
        
        try:
            result = subprocess.run("conda --version", shell=True, 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                conda_version = result.stdout.strip()
                print(f"✓ {conda_version}")
                
                # Check conda-forge channel
                result = subprocess.run("conda config --show channels", shell=True,
                                      capture_output=True, text=True, timeout=10)
                if "conda-forge" in result.stdout:
                    print("✓ conda-forge channel available")
                else:
                    self.warnings.append("conda-forge channel not configured. Run: conda config --add channels conda-forge")
                
                return True
            else:
                self.errors.append("conda command failed")
                return False
                
        except subprocess.TimeoutExpired:
            self.errors.append("conda command timed out")
            return False
        except FileNotFoundError:
            self.errors.append("conda not found in PATH")
            return False
    
    def check_environment_exists(self, env_name: str) -> bool:
        """Check if conda environment exists and is properly configured."""
        print(f"Checking conda environment: {env_name}")
        
        try:
            result = subprocess.run(f"conda env list", shell=True,
                                  capture_output=True, text=True, timeout=10)
            
            if env_name in result.stdout:
                print(f"✓ Environment {env_name} exists")
                
                # Check Python version in environment
                try:
                    result = subprocess.run(f"conda run -n {env_name} python --version", 
                                          shell=True, capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        python_version = result.stdout.strip()
                        print(f"✓ {python_version} in environment")
                    else:
                        self.warnings.append(f"Could not check Python version in {env_name}")
                except:
                    self.warnings.append(f"Could not access environment {env_name}")
                
                return True
            else:
                self.errors.append(f"Environment {env_name} does not exist")
                return False
                
        except Exception as e:
            self.errors.append(f"Error checking environment: {e}")
            return False
    
    def check_tensorflow_installation(self, env_name: str) -> bool:
        """Check TensorFlow installation in conda environment."""
        print("Checking TensorFlow installation...")
        
        try:
            # Check if TensorFlow is installed
            result = subprocess.run(f"conda run -n {env_name} python -c \"import tensorflow as tf; print(f'TensorFlow {tf.__version__}')\"",
                                  shell=True, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                tf_version = result.stdout.strip()
                print(f"✓ {tf_version}")
                
                # Check GPU support
                result = subprocess.run(f"conda run -n {env_name} python -c \"import tensorflow as tf; print('GPU available:', tf.config.list_physical_devices('GPU'))\"",
                                      shell=True, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    gpu_info = result.stdout.strip()
                    if "GPU available: []" in gpu_info:
                        self.warnings.append("TensorFlow installed but no GPU detected")
                    else:
                        print("✓ GPU support detected")
                
                return True
            else:
                self.errors.append(f"TensorFlow import failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.errors.append(f"Error checking TensorFlow: {e}")
            return False
    
    def check_cuda_installation(self) -> bool:
        """Check CUDA installation and compatibility."""
        print("Checking CUDA installation...")
        
        try:
            # Check nvidia-smi
            result = subprocess.run("nvidia-smi", shell=True,
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Extract CUDA version from nvidia-smi output
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'CUDA Version:' in line:
                        cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                        print(f"✓ CUDA {cuda_version} detected")
                        break
                
                # Check driver version
                for line in lines:
                    if 'Driver Version:' in line:
                        driver_version = line.split('Driver Version:')[1].strip().split()[0]
                        print(f"✓ NVIDIA Driver {driver_version}")
                        break
                
                return True
            else:
                self.warnings.append("nvidia-smi not available - GPU may not be accessible")
                return False
                
        except FileNotFoundError:
            self.warnings.append("nvidia-smi not found - NVIDIA drivers may not be installed")
            return False
        except Exception as e:
            self.warnings.append(f"Error checking CUDA: {e}")
            return False
    
    def check_dataset_availability(self, dataset_path: str) -> Dict[str, bool]:
        """Check dataset availability and structure."""
        print(f"Checking dataset at: {dataset_path}")
        
        dataset_path = Path(dataset_path)
        dataset_status = {}
        
        expected_datasets = {
            'doc3d': 'Full Doc3D dataset',
            'doc3d_100': 'Doc3D 100 samples',
            'doc3d_1000': 'Doc3D 1000 samples',
            'input_crop': 'Input crop images'
        }
        
        for dataset_name, description in expected_datasets.items():
            dataset_dir = dataset_path / dataset_name
            if dataset_dir.exists():
                # Count files to estimate dataset size
                try:
                    file_count = len(list(dataset_dir.rglob('*')))
                    print(f"✓ {description}: {file_count} files")
                    dataset_status[dataset_name] = True
                except:
                    print(f"✓ {description}: Available (could not count files)")
                    dataset_status[dataset_name] = True
            else:
                print(f"❌ {description}: Not found")
                dataset_status[dataset_name] = False
        
        return dataset_status
    
    def check_disk_space(self, paths: List[str]) -> bool:
        """Check available disk space."""
        print("Checking disk space...")
        
        try:
            import shutil
            
            for path in paths:
                if Path(path).exists():
                    total, used, free = shutil.disk_usage(path)
                    free_gb = free / (1024**3)
                    total_gb = total / (1024**3)
                    
                    print(f"✓ {path}: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
                    
                    if free_gb < 10:
                        self.warnings.append(f"Low disk space at {path}: {free_gb:.1f} GB free")
                    elif free_gb < 50:
                        self.warnings.append(f"Limited disk space at {path}: {free_gb:.1f} GB free (recommend 100GB+ for training)")
            
            return True
            
        except Exception as e:
            self.warnings.append(f"Could not check disk space: {e}")
            return False
    
    def generate_troubleshooting_guide(self) -> str:
        """Generate troubleshooting guide based on validation results."""
        guide = []
        
        if self.errors:
            guide.append("=== CRITICAL ISSUES ===")
            for error in self.errors:
                guide.append(f"❌ {error}")
            
            guide.append("\n=== SOLUTIONS ===")
            
            # Conda issues
            if any("conda" in error.lower() for error in self.errors):
                guide.append("Conda Issues:")
                guide.append("1. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html")
                guide.append("2. Add conda to PATH: export PATH=\"$HOME/miniconda3/bin:$PATH\"")
                guide.append("3. Restart terminal and try again")
            
            # Environment issues
            if any("environment" in error.lower() for error in self.errors):
                guide.append("\nEnvironment Issues:")
                guide.append("1. Create environment: python tensorflow/setup_env.py")
                guide.append("2. Or manually: conda create -n dewarpnet_tf_kiro_full python=3.9 -y")
            
            # TensorFlow issues
            if any("tensorflow" in error.lower() for error in self.errors):
                guide.append("\nTensorFlow Issues:")
                guide.append("1. Install TensorFlow: conda run -n dewarpnet_tf_kiro_full pip install tensorflow>=2.10.0")
                guide.append("2. For GPU support: conda install -n dewarpnet_tf_kiro_full cudatoolkit cudnn -c conda-forge")
        
        if self.warnings:
            guide.append("\n=== WARNINGS ===")
            for warning in self.warnings:
                guide.append(f"⚠️  {warning}")
        
        return "\n".join(guide)
    
    def validate_complete_environment(self, env_name: str = "dewarpnet_tf_kiro_full", 
                                    dataset_path: str = "/home/argus/Workspace/dataset") -> Dict:
        """Run complete environment validation."""
        print("=== TensorFlow DewarpNet Environment Validation ===\n")
        
        validation_results = {
            'system_requirements': self.check_system_requirements(),
            'conda_installation': self.check_conda_installation(),
            'environment_exists': self.check_environment_exists(env_name),
            'tensorflow_installation': False,
            'cuda_installation': self.check_cuda_installation(),
            'dataset_availability': self.check_dataset_availability(dataset_path),
            'disk_space': self.check_disk_space([str(Path.cwd()), dataset_path])
        }
        
        # Only check TensorFlow if environment exists
        if validation_results['environment_exists']:
            validation_results['tensorflow_installation'] = self.check_tensorflow_installation(env_name)
        
        print("\n=== Validation Summary ===")
        
        # Critical checks
        critical_checks = ['system_requirements', 'conda_installation', 'environment_exists', 'tensorflow_installation']
        critical_passed = all(validation_results.get(check, False) for check in critical_checks)
        
        for check, result in validation_results.items():
            if check == 'dataset_availability':
                available_datasets = sum(result.values())
                total_datasets = len(result)
                status = "✓" if available_datasets > 0 else "❌"
                print(f"{status} {check}: {available_datasets}/{total_datasets} datasets available")
            else:
                status = "✓" if result else "❌"
                print(f"{status} {check}: {result}")
        
        print(f"\nOverall Status: {'✓ READY' if critical_passed else '❌ NEEDS ATTENTION'}")
        
        # Generate troubleshooting guide if needed
        if self.errors or self.warnings:
            print("\n" + self.generate_troubleshooting_guide())
        
        return {
            'validation_results': validation_results,
            'errors': self.errors,
            'warnings': self.warnings,
            'overall_status': critical_passed
        }


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate TensorFlow DewarpNet environment")
    parser.add_argument("--env-name", default="dewarpnet_tf_kiro_full",
                       help="Conda environment name to validate")
    parser.add_argument("--dataset-path", default="/home/argus/Workspace/dataset",
                       help="Path to dataset directory")
    parser.add_argument("--output", help="Save validation results to JSON file")
    
    args = parser.parse_args()
    
    validator = EnvironmentValidator()
    results = validator.validate_complete_environment(args.env_name, args.dataset_path)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nValidation results saved to: {args.output}")


if __name__ == "__main__":
    main()