#!/usr/bin/env python3
"""
Dependency validation and installation utilities.
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


class DependencyValidator:
    """Validate and manage Python dependencies for TensorFlow DewarpNet."""
    
    def __init__(self, requirements_file: str = "tensorflow/requirements_tf.txt"):
        self.requirements_file = Path(requirements_file)
        self.required_packages = self._parse_requirements()
        self.validation_results = {}
        
    def _parse_requirements(self) -> Dict[str, str]:
        """Parse requirements.txt file."""
        packages = {}
        
        if not self.requirements_file.exists():
            print(f"Warning: Requirements file not found: {self.requirements_file}")
            return packages
        
        with open(self.requirements_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '>=' in line:
                        name, version = line.split('>=')
                        packages[name.strip()] = version.strip()
                    elif '==' in line:
                        name, version = line.split('==')
                        packages[name.strip()] = version.strip()
                    else:
                        packages[line.strip()] = 'any'
        
        return packages
    
    def check_package_installation(self, package_name: str) -> Tuple[bool, Optional[str]]:
        """Check if a package is installed and get its version."""
        try:
            # Handle special package name mappings
            import_name = package_name
            if package_name == 'opencv-python':
                import_name = 'cv2'
            elif package_name == 'Pillow':
                import_name = 'PIL'
            elif package_name == 'scikit-image':
                import_name = 'skimage'
            
            module = importlib.import_module(import_name)
            
            # Try to get version
            version = None
            for attr in ['__version__', 'version', 'VERSION']:
                if hasattr(module, attr):
                    version = getattr(module, attr)
                    if callable(version):
                        version = version()
                    break
            
            return True, str(version) if version else 'unknown'
            
        except ImportError:
            return False, None
    
    def validate_all_dependencies(self) -> Dict[str, Dict]:
        """Validate all required dependencies."""
        print("Validating dependencies...")
        
        results = {}
        
        for package, required_version in self.required_packages.items():
            installed, current_version = self.check_package_installation(package)
            
            status = {
                'installed': installed,
                'required_version': required_version,
                'current_version': current_version,
                'compatible': True
            }
            
            if installed and current_version and required_version != 'any':
                # Simple version comparison (works for most cases)
                try:
                    from packaging import version
                    if version.parse(current_version) < version.parse(required_version):
                        status['compatible'] = False
                except:
                    # Fallback to string comparison if packaging not available
                    if current_version < required_version:
                        status['compatible'] = False
            
            results[package] = status
            
            # Print status
            if installed:
                if status['compatible']:
                    print(f"✓ {package}: {current_version}")
                else:
                    print(f"⚠️  {package}: {current_version} (requires {required_version})")
            else:
                print(f"❌ {package}: Not installed")
        
        self.validation_results = results
        return results
    
    def get_missing_packages(self) -> List[str]:
        """Get list of missing packages."""
        if not self.validation_results:
            self.validate_all_dependencies()
        
        return [pkg for pkg, status in self.validation_results.items() 
                if not status['installed']]
    
    def get_outdated_packages(self) -> List[str]:
        """Get list of outdated packages."""
        if not self.validation_results:
            self.validate_all_dependencies()
        
        return [pkg for pkg, status in self.validation_results.items() 
                if status['installed'] and not status['compatible']]
    
    def install_missing_packages(self, use_conda: bool = False) -> Dict[str, bool]:
        """Install missing packages."""
        missing = self.get_missing_packages()
        
        if not missing:
            print("✓ All packages are installed")
            return {}
        
        print(f"Installing {len(missing)} missing packages...")
        
        installation_results = {}
        
        for package in missing:
            print(f"Installing {package}...")
            
            try:
                if use_conda:
                    # Try conda first, fall back to pip
                    result = subprocess.run(f"conda install {package} -y", 
                                          shell=True, capture_output=True, text=True)
                    if result.returncode != 0:
                        result = subprocess.run(f"pip install {package}", 
                                              shell=True, capture_output=True, text=True)
                else:
                    result = subprocess.run(f"pip install {package}", 
                                          shell=True, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✓ {package} installed successfully")
                    installation_results[package] = True
                else:
                    print(f"❌ Failed to install {package}: {result.stderr}")
                    installation_results[package] = False
                    
            except Exception as e:
                print(f"❌ Error installing {package}: {e}")
                installation_results[package] = False
        
        return installation_results
    
    def upgrade_outdated_packages(self) -> Dict[str, bool]:
        """Upgrade outdated packages."""
        outdated = self.get_outdated_packages()
        
        if not outdated:
            print("✓ All packages are up to date")
            return {}
        
        print(f"Upgrading {len(outdated)} outdated packages...")
        
        upgrade_results = {}
        
        for package in outdated:
            required_version = self.required_packages[package]
            print(f"Upgrading {package} to {required_version}...")
            
            try:
                result = subprocess.run(f"pip install {package}>={required_version}", 
                                      shell=True, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✓ {package} upgraded successfully")
                    upgrade_results[package] = True
                else:
                    print(f"❌ Failed to upgrade {package}: {result.stderr}")
                    upgrade_results[package] = False
                    
            except Exception as e:
                print(f"❌ Error upgrading {package}: {e}")
                upgrade_results[package] = False
        
        return upgrade_results
    
    def check_tensorflow_gpu_support(self) -> Dict[str, any]:
        """Check TensorFlow GPU support specifically."""
        print("Checking TensorFlow GPU support...")
        
        gpu_info = {
            'tensorflow_installed': False,
            'tensorflow_version': None,
            'gpu_support_built': False,
            'gpus_available': 0,
            'gpu_devices': []
        }
        
        try:
            import tensorflow as tf
            gpu_info['tensorflow_installed'] = True
            gpu_info['tensorflow_version'] = tf.__version__
            
            # Check if built with CUDA
            gpu_info['gpu_support_built'] = tf.test.is_built_with_cuda()
            
            # Check available GPUs
            gpus = tf.config.list_physical_devices('GPU')
            gpu_info['gpus_available'] = len(gpus)
            gpu_info['gpu_devices'] = [gpu.name for gpu in gpus]
            
            print(f"✓ TensorFlow {tf.__version__}")
            print(f"✓ CUDA support built: {gpu_info['gpu_support_built']}")
            print(f"✓ GPUs available: {gpu_info['gpus_available']}")
            
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
                
        except ImportError:
            print("❌ TensorFlow not installed")
        except Exception as e:
            print(f"❌ Error checking TensorFlow: {e}")
        
        return gpu_info
    
    def check_cuda_packages(self) -> Dict[str, any]:
        """Check CUDA-related packages."""
        print("Checking CUDA packages...")
        
        cuda_packages = ['cudatoolkit', 'cudnn']
        cuda_info = {}
        
        for package in cuda_packages:
            try:
                result = subprocess.run(f"conda list {package}", 
                                      shell=True, capture_output=True, text=True)
                
                if result.returncode == 0 and package in result.stdout:
                    # Extract version from conda list output
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if package in line and not line.startswith('#'):
                            parts = line.split()
                            if len(parts) >= 2:
                                version = parts[1]
                                cuda_info[package] = {
                                    'installed': True,
                                    'version': version
                                }
                                print(f"✓ {package}: {version}")
                                break
                else:
                    cuda_info[package] = {'installed': False, 'version': None}
                    print(f"❌ {package}: Not installed")
                    
            except Exception as e:
                cuda_info[package] = {'installed': False, 'error': str(e)}
                print(f"❌ Error checking {package}: {e}")
        
        return cuda_info
    
    def generate_dependency_report(self) -> Dict:
        """Generate comprehensive dependency report."""
        print("=== Dependency Validation Report ===\n")
        
        # Validate all dependencies
        dep_results = self.validate_all_dependencies()
        
        # Check TensorFlow GPU support
        tf_gpu_info = self.check_tensorflow_gpu_support()
        
        # Check CUDA packages
        cuda_info = self.check_cuda_packages()
        
        report = {
            'dependencies': dep_results,
            'tensorflow_gpu': tf_gpu_info,
            'cuda_packages': cuda_info,
            'summary': {
                'total_packages': len(dep_results),
                'installed_packages': sum(1 for s in dep_results.values() if s['installed']),
                'missing_packages': len(self.get_missing_packages()),
                'outdated_packages': len(self.get_outdated_packages()),
                'tensorflow_ready': tf_gpu_info['tensorflow_installed'],
                'gpu_ready': tf_gpu_info['gpus_available'] > 0
            }
        }
        
        # Print summary
        print("\n=== Summary ===")
        summary = report['summary']
        print(f"Total packages: {summary['total_packages']}")
        print(f"Installed: {summary['installed_packages']}")
        print(f"Missing: {summary['missing_packages']}")
        print(f"Outdated: {summary['outdated_packages']}")
        print(f"TensorFlow ready: {summary['tensorflow_ready']}")
        print(f"GPU ready: {summary['gpu_ready']}")
        
        return report
    
    def fix_all_dependencies(self, use_conda: bool = False) -> Dict:
        """Fix all dependency issues."""
        print("=== Fixing All Dependencies ===\n")
        
        # Install missing packages
        install_results = self.install_missing_packages(use_conda)
        
        # Upgrade outdated packages
        upgrade_results = self.upgrade_outdated_packages()
        
        # Re-validate after fixes
        print("\nRe-validating dependencies...")
        final_results = self.validate_all_dependencies()
        
        return {
            'installations': install_results,
            'upgrades': upgrade_results,
            'final_validation': final_results
        }


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate and fix dependencies")
    parser.add_argument("--requirements", default="tensorflow/requirements_tf.txt",
                       help="Path to requirements file")
    parser.add_argument("--fix", action="store_true",
                       help="Automatically fix missing/outdated packages")
    parser.add_argument("--use-conda", action="store_true",
                       help="Use conda for package installation")
    parser.add_argument("--report", help="Save report to JSON file")
    
    args = parser.parse_args()
    
    validator = DependencyValidator(args.requirements)
    
    if args.fix:
        results = validator.fix_all_dependencies(args.use_conda)
        print(f"\nFix results: {results}")
    else:
        report = validator.generate_dependency_report()
        
        if args.report:
            with open(args.report, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\nReport saved to: {args.report}")


if __name__ == "__main__":
    main()