#!/usr/bin/env python3
"""
Dataset preparation utilities for TensorFlow DewarpNet.
Handles dataset size selection, preparation, and validation.
"""

import os
import sys
import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse


class DatasetPreparator:
    """Prepare datasets for training and testing."""
    
    def __init__(self, base_path: str = "/home/argus/Workspace/dataset"):
        self.base_path = Path(base_path)
        self.project_root = Path.cwd()
        self.data_dir = self.project_root / "tensorflow" / "data"
        
        # Dataset size configurations
        self.size_configs = {
            'doc3d_10': {
                'description': 'Tiny dataset for quick testing (10 samples)',
                'sample_count': 10,
                'source': 'doc3d_100',
                'use_case': 'Unit testing and quick validation'
            },
            'doc3d_50': {
                'description': 'Small dataset for development (50 samples)', 
                'sample_count': 50,
                'source': 'doc3d_100',
                'use_case': 'Development and debugging'
            },
            'doc3d_100': {
                'description': 'Standard test dataset (100 samples)',
                'sample_count': 100,
                'source': 'doc3d',
                'use_case': 'Initial validation and testing'
            },
            'doc3d_500': {
                'description': 'Medium dataset for experiments (500 samples)',
                'sample_count': 500,
                'source': 'doc3d_1000',
                'use_case': 'Medium-scale experiments'
            },
            'doc3d_1000': {
                'description': 'Large test dataset (1000 samples)',
                'sample_count': 1000,
                'source': 'doc3d',
                'use_case': 'Large-scale validation'
            },
            'doc3d': {
                'description': 'Full dataset (all samples)',
                'sample_count': -1,  # All samples
                'source': None,
                'use_case': 'Production training'
            }
        }
    
    def list_available_datasets(self) -> Dict[str, Dict]:
        """List all available datasets with their status."""
        print("=== Available Datasets ===\n")
        
        dataset_status = {}
        
        for dataset_name, config in self.size_configs.items():
            dataset_path = self.base_path / dataset_name
            link_path = self.data_dir / dataset_name
            
            status = {
                'config': config,
                'source_exists': dataset_path.exists(),
                'link_exists': link_path.exists() or link_path.is_symlink(),
                'file_count': 0,
                'size_mb': 0
            }
            
            # Get file count and size if dataset exists
            if dataset_path.exists():
                try:
                    files = list(dataset_path.rglob('*'))
                    files = [f for f in files if f.is_file()]
                    status['file_count'] = len(files)
                    status['size_mb'] = sum(f.stat().st_size for f in files) / (1024 * 1024)
                except:
                    pass
            
            dataset_status[dataset_name] = status
            
            # Print status
            source_icon = "✓" if status['source_exists'] else "❌"
            link_icon = "✓" if status['link_exists'] else "❌"
            
            print(f"{dataset_name}:")
            print(f"  Description: {config['description']}")
            print(f"  Use case: {config['use_case']}")
            print(f"  Source exists: {source_icon}")
            print(f"  Link exists: {link_icon}")
            if status['file_count'] > 0:
                print(f"  Files: {status['file_count']:,}")
                print(f"  Size: {status['size_mb']:.1f} MB")
            print()
        
        return dataset_status
    
    def prepare_dataset_size(self, size_name: str, force: bool = False) -> bool:
        """Prepare a specific dataset size."""
        if size_name not in self.size_configs:
            print(f"❌ Unknown dataset size: {size_name}")
            print(f"Available sizes: {list(self.size_configs.keys())}")
            return False
        
        config = self.size_configs[size_name]
        print(f"Preparing dataset: {size_name}")
        print(f"Description: {config['description']}")
        
        dataset_path = self.base_path / size_name
        
        # Check if dataset already exists
        if dataset_path.exists() and not force:
            print(f"✓ Dataset {size_name} already exists")
            return self._create_symbolic_link(size_name)
        
        # For full dataset, just create link
        if size_name == 'doc3d':
            return self._create_symbolic_link(size_name)
        
        # Create subset dataset
        source_name = config['source']
        if not source_name:
            print(f"❌ No source specified for {size_name}")
            return False
        
        source_path = self.base_path / source_name
        if not source_path.exists():
            print(f"❌ Source dataset does not exist: {source_path}")
            print(f"Please ensure {source_name} is available first")
            return False
        
        return self._create_dataset_subset(source_name, size_name, config['sample_count'], force)
    
    def _create_dataset_subset(self, source_name: str, target_name: str, 
                              sample_count: int, force: bool = False) -> bool:
        """Create a subset of a dataset."""
        from .dataset_manager import DatasetManager
        
        print(f"Creating {target_name} from {source_name} ({sample_count} samples)...")
        
        manager = DatasetManager(str(self.base_path))
        success = manager.create_dataset_subset(source_name, target_name, sample_count)
        
        if success:
            return self._create_symbolic_link(target_name)
        
        return False
    
    def _create_symbolic_link(self, dataset_name: str) -> bool:
        """Create symbolic link for a dataset."""
        source_path = self.base_path / dataset_name
        link_path = self.data_dir / dataset_name
        
        if not source_path.exists():
            print(f"❌ Source dataset does not exist: {source_path}")
            return False
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Remove existing link if it exists
        if link_path.exists() or link_path.is_symlink():
            if link_path.is_symlink():
                link_path.unlink()
            else:
                shutil.rmtree(link_path)
        
        try:
            # Create symbolic link
            if os.name == 'nt':  # Windows
                import subprocess
                result = subprocess.run(
                    f'mklink /D "{link_path}" "{source_path}"',
                    shell=True, capture_output=True, text=True
                )
                success = result.returncode == 0
            else:  # Unix-like systems
                link_path.symlink_to(source_path)
                success = True
            
            if success:
                print(f"✓ Created symbolic link: {dataset_name}")
                return True
            else:
                print(f"❌ Failed to create symbolic link: {dataset_name}")
                return False
                
        except Exception as e:
            print(f"❌ Error creating symbolic link: {e}")
            return False
    
    def validate_dataset_for_training(self, dataset_name: str) -> Dict[str, any]:
        """Validate that a dataset is ready for training."""
        print(f"Validating {dataset_name} for training...")
        
        validation = {
            'dataset_name': dataset_name,
            'ready_for_training': False,
            'issues': [],
            'file_counts': {},
            'recommendations': []
        }
        
        # Check if dataset exists in project data directory
        dataset_path = self.data_dir / dataset_name
        if not (dataset_path.exists() or dataset_path.is_symlink()):
            validation['issues'].append(f"Dataset not found in project data directory: {dataset_path}")
            validation['recommendations'].append(f"Run: python tensorflow/utils/dataset_prep.py --prepare {dataset_name}")
            return validation
        
        # Resolve actual dataset path
        if dataset_path.is_symlink():
            actual_path = dataset_path.resolve()
        else:
            actual_path = dataset_path
        
        if not actual_path.exists():
            validation['issues'].append(f"Dataset link is broken: {dataset_path} -> {actual_path}")
            return validation
        
        # Check required subdirectories for training
        required_subdirs = ['img', 'wc']  # Minimum for world coordinate training
        optional_subdirs = ['bm', 'albedo']  # For backward mapping training
        
        for subdir in required_subdirs:
            subdir_path = actual_path / subdir
            if subdir_path.exists():
                file_count = len([f for f in subdir_path.rglob('*') if f.is_file()])
                validation['file_counts'][subdir] = file_count
                
                if file_count == 0:
                    validation['issues'].append(f"Required subdirectory {subdir} is empty")
            else:
                validation['issues'].append(f"Required subdirectory missing: {subdir}")
                validation['file_counts'][subdir] = 0
        
        for subdir in optional_subdirs:
            subdir_path = actual_path / subdir
            if subdir_path.exists():
                file_count = len([f for f in subdir_path.rglob('*') if f.is_file()])
                validation['file_counts'][subdir] = file_count
            else:
                validation['file_counts'][subdir] = 0
        
        # Check file count consistency
        img_count = validation['file_counts'].get('img', 0)
        wc_count = validation['file_counts'].get('wc', 0)
        
        if img_count != wc_count:
            validation['issues'].append(f"File count mismatch: img={img_count}, wc={wc_count}")
        
        # Check minimum file count
        min_files = 10 if dataset_name.endswith('_10') else 50
        if img_count < min_files:
            validation['issues'].append(f"Too few samples: {img_count} (minimum {min_files})")
        
        # Generate recommendations
        if validation['file_counts'].get('bm', 0) == 0:
            validation['recommendations'].append("Backward mapping training not available (missing 'bm' directory)")
        
        if validation['file_counts'].get('albedo', 0) == 0:
            validation['recommendations'].append("Albedo data not available (missing 'albedo' directory)")
        
        # Determine if ready for training
        validation['ready_for_training'] = len(validation['issues']) == 0
        
        # Print results
        if validation['ready_for_training']:
            print(f"✓ {dataset_name} is ready for training")
            print(f"  Images: {validation['file_counts']['img']:,}")
            print(f"  World coordinates: {validation['file_counts']['wc']:,}")
            if validation['file_counts'].get('bm', 0) > 0:
                print(f"  Backward mapping: {validation['file_counts']['bm']:,}")
        else:
            print(f"❌ {dataset_name} has issues:")
            for issue in validation['issues']:
                print(f"  - {issue}")
        
        if validation['recommendations']:
            print("Recommendations:")
            for rec in validation['recommendations']:
                print(f"  - {rec}")
        
        return validation
    
    def setup_recommended_datasets(self) -> Dict[str, bool]:
        """Set up recommended datasets for development."""
        print("=== Setting up recommended datasets ===\n")
        
        # Recommended setup: doc3d_10 for testing, doc3d_100 for validation
        recommended = ['doc3d_10', 'doc3d_100']
        results = {}
        
        for dataset_name in recommended:
            print(f"Setting up {dataset_name}...")
            results[dataset_name] = self.prepare_dataset_size(dataset_name)
            print()
        
        # Summary
        successful = sum(results.values())
        total = len(results)
        
        print(f"=== Setup Summary ===")
        print(f"Successfully set up: {successful}/{total} datasets")
        
        for dataset_name, success in results.items():
            status = "✓" if success else "❌"
            print(f"{status} {dataset_name}")
        
        if successful == total:
            print("\n🎉 Recommended datasets are ready!")
            print("Next steps:")
            print("1. Validate setup: python tensorflow/utils/test_setup.py")
            print("2. Run training: python tensorflow/training/train_wc.py --dataset doc3d_10")
        
        return results
    
    def interactive_dataset_selection(self) -> Optional[str]:
        """Interactive dataset selection for users."""
        print("=== Interactive Dataset Selection ===\n")
        
        # Show available datasets
        dataset_status = self.list_available_datasets()
        
        print("Select a dataset size for your use case:")
        print("1. Quick testing (doc3d_10) - 10 samples")
        print("2. Development (doc3d_50) - 50 samples") 
        print("3. Standard validation (doc3d_100) - 100 samples")
        print("4. Medium experiments (doc3d_500) - 500 samples")
        print("5. Large validation (doc3d_1000) - 1000 samples")
        print("6. Full training (doc3d) - All samples")
        print("7. Show detailed status")
        print("8. Exit")
        
        try:
            choice = input("\nEnter your choice (1-8): ").strip()
            
            choices = {
                '1': 'doc3d_10',
                '2': 'doc3d_50', 
                '3': 'doc3d_100',
                '4': 'doc3d_500',
                '5': 'doc3d_1000',
                '6': 'doc3d'
            }
            
            if choice in choices:
                dataset_name = choices[choice]
                print(f"\nSelected: {dataset_name}")
                
                # Ask for confirmation
                config = self.size_configs[dataset_name]
                print(f"Description: {config['description']}")
                print(f"Use case: {config['use_case']}")
                
                confirm = input("Proceed with setup? (y/n): ").strip().lower()
                if confirm == 'y':
                    success = self.prepare_dataset_size(dataset_name)
                    if success:
                        self.validate_dataset_for_training(dataset_name)
                    return dataset_name
                else:
                    print("Setup cancelled.")
                    return None
            
            elif choice == '7':
                # Show detailed status
                print("\n" + "="*50)
                self.list_available_datasets()
                return self.interactive_dataset_selection()  # Recurse
            
            elif choice == '8':
                print("Exiting...")
                return None
            
            else:
                print("Invalid choice. Please try again.")
                return self.interactive_dataset_selection()  # Recurse
                
        except KeyboardInterrupt:
            print("\nSelection cancelled.")
            return None
    
    def cleanup_datasets(self, keep_sizes: List[str] = None) -> Dict[str, bool]:
        """Clean up unused dataset sizes."""
        if keep_sizes is None:
            keep_sizes = ['doc3d_100', 'doc3d']  # Keep standard sizes
        
        print(f"Cleaning up datasets (keeping: {keep_sizes})...")
        
        cleanup_results = {}
        
        for dataset_name in self.size_configs.keys():
            if dataset_name in keep_sizes:
                continue
            
            dataset_path = self.base_path / dataset_name
            link_path = self.data_dir / dataset_name
            
            removed = False
            
            # Remove dataset directory
            if dataset_path.exists() and dataset_path != self.base_path / 'doc3d':
                try:
                    shutil.rmtree(dataset_path)
                    print(f"✓ Removed dataset: {dataset_path}")
                    removed = True
                except Exception as e:
                    print(f"❌ Error removing {dataset_path}: {e}")
            
            # Remove symbolic link
            if link_path.exists() or link_path.is_symlink():
                try:
                    if link_path.is_symlink():
                        link_path.unlink()
                    else:
                        shutil.rmtree(link_path)
                    print(f"✓ Removed link: {link_path}")
                    removed = True
                except Exception as e:
                    print(f"❌ Error removing {link_path}: {e}")
            
            cleanup_results[dataset_name] = removed
        
        return cleanup_results


def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(description="Dataset preparation utilities")
    parser.add_argument("--base-path", default="/home/argus/Workspace/dataset",
                       help="Base dataset path")
    parser.add_argument("--list", action="store_true",
                       help="List available datasets")
    parser.add_argument("--prepare", metavar="SIZE",
                       help="Prepare specific dataset size")
    parser.add_argument("--force", action="store_true",
                       help="Force recreate dataset")
    parser.add_argument("--validate", metavar="SIZE",
                       help="Validate dataset for training")
    parser.add_argument("--setup-recommended", action="store_true",
                       help="Set up recommended datasets")
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive dataset selection")
    parser.add_argument("--cleanup", nargs="*", metavar="KEEP",
                       help="Clean up datasets (specify sizes to keep)")
    
    args = parser.parse_args()
    
    prep = DatasetPreparator(args.base_path)
    
    if args.list:
        prep.list_available_datasets()
    
    elif args.prepare:
        prep.prepare_dataset_size(args.prepare, args.force)
    
    elif args.validate:
        prep.validate_dataset_for_training(args.validate)
    
    elif args.setup_recommended:
        prep.setup_recommended_datasets()
    
    elif args.interactive:
        prep.interactive_dataset_selection()
    
    elif args.cleanup is not None:
        keep_sizes = args.cleanup if args.cleanup else ['doc3d_100', 'doc3d']
        prep.cleanup_datasets(keep_sizes)
    
    else:
        # Default: show available datasets
        prep.list_available_datasets()


if __name__ == "__main__":
    main()