#!/usr/bin/env python3
"""
Dataset configuration and management utilities for TensorFlow DewarpNet.
"""

import os
import sys
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import random
import argparse


class DatasetManager:
    """Manage dataset configuration, validation, and preparation."""
    
    def __init__(self, base_dataset_path: str = "/home/argus/Workspace/dataset"):
        self.base_path = Path(base_dataset_path)
        self.project_data_dir = Path("tensorflow/data")
        
        # Dataset configurations
        self.dataset_configs = {
            'doc3d': {
                'description': 'Full Doc3D dataset',
                'expected_size': '~100GB',
                'min_files': 100000,
                'subdirs': ['img', 'wc', 'bm', 'albedo']
            },
            'doc3d_1000': {
                'description': 'Doc3D 1000 samples',
                'expected_size': '~10GB', 
                'min_files': 4000,
                'subdirs': ['img', 'wc', 'bm', 'albedo']
            },
            'doc3d_100': {
                'description': 'Doc3D 100 samples',
                'expected_size': '~1GB',
                'min_files': 400,
                'subdirs': ['img', 'wc', 'bm', 'albedo']
            },
            'input_crop': {
                'description': 'Input crop images for inference',
                'expected_size': '~100MB',
                'min_files': 10,
                'subdirs': []
            }
        }
    
    def validate_dataset_structure(self, dataset_name: str) -> Dict[str, any]:
        """Validate dataset structure and integrity."""
        print(f"Validating dataset: {dataset_name}")
        
        dataset_path = self.base_path / dataset_name
        config = self.dataset_configs.get(dataset_name, {})
        
        validation_result = {
            'dataset_name': dataset_name,
            'path': str(dataset_path),
            'exists': dataset_path.exists(),
            'is_directory': dataset_path.is_dir() if dataset_path.exists() else False,
            'file_count': 0,
            'size_bytes': 0,
            'subdirectories': {},
            'sample_files': [],
            'issues': [],
            'status': 'unknown'
        }
        
        if not dataset_path.exists():
            validation_result['issues'].append(f"Dataset directory does not exist: {dataset_path}")
            validation_result['status'] = 'missing'
            return validation_result
        
        if not dataset_path.is_dir():
            validation_result['issues'].append(f"Path exists but is not a directory: {dataset_path}")
            validation_result['status'] = 'invalid'
            return validation_result
        
        # Count files and calculate size
        try:
            total_files = 0
            total_size = 0
            
            for file_path in dataset_path.rglob('*'):
                if file_path.is_file():
                    total_files += 1
                    total_size += file_path.stat().st_size
                    
                    # Collect sample files
                    if len(validation_result['sample_files']) < 5:
                        validation_result['sample_files'].append(str(file_path.relative_to(dataset_path)))
            
            validation_result['file_count'] = total_files
            validation_result['size_bytes'] = total_size
            
            # Check subdirectories
            expected_subdirs = config.get('subdirs', [])
            for subdir in expected_subdirs:
                subdir_path = dataset_path / subdir
                validation_result['subdirectories'][subdir] = {
                    'exists': subdir_path.exists(),
                    'file_count': len(list(subdir_path.rglob('*'))) if subdir_path.exists() else 0
                }
                
                if not subdir_path.exists():
                    validation_result['issues'].append(f"Expected subdirectory missing: {subdir}")
            
            # Check minimum file count
            min_files = config.get('min_files', 0)
            if total_files < min_files:
                validation_result['issues'].append(f"File count ({total_files}) below minimum ({min_files})")
            
            # Determine status
            if validation_result['issues']:
                validation_result['status'] = 'incomplete'
            else:
                validation_result['status'] = 'valid'
            
            print(f"✓ {dataset_name}: {total_files} files, {self._format_size(total_size)}")
            
        except Exception as e:
            validation_result['issues'].append(f"Error during validation: {str(e)}")
            validation_result['status'] = 'error'
            print(f"❌ Error validating {dataset_name}: {e}")
        
        return validation_result
    
    def validate_all_datasets(self) -> Dict[str, Dict]:
        """Validate all configured datasets."""
        print("=== Dataset Validation ===\n")
        
        results = {}
        
        for dataset_name in self.dataset_configs.keys():
            results[dataset_name] = self.validate_dataset_structure(dataset_name)
            print()
        
        # Summary
        valid_count = sum(1 for r in results.values() if r['status'] == 'valid')
        total_count = len(results)
        
        print(f"=== Validation Summary ===")
        print(f"Valid datasets: {valid_count}/{total_count}")
        
        for name, result in results.items():
            status_icon = {
                'valid': '✓',
                'incomplete': '⚠️',
                'missing': '❌',
                'invalid': '❌',
                'error': '❌',
                'unknown': '?'
            }.get(result['status'], '?')
            
            print(f"{status_icon} {name}: {result['status']}")
            if result['issues']:
                for issue in result['issues']:
                    print(f"    - {issue}")
        
        return results
    
    def create_symbolic_links(self, force: bool = False) -> Dict[str, bool]:
        """Create symbolic links for datasets."""
        print("Creating symbolic links...")
        
        # Ensure project data directory exists
        self.project_data_dir.mkdir(parents=True, exist_ok=True)
        
        link_results = {}
        
        for dataset_name in self.dataset_configs.keys():
            source_path = self.base_path / dataset_name
            link_path = self.project_data_dir / dataset_name
            
            try:
                # Check if source exists
                if not source_path.exists():
                    print(f"⚠️  Skipping {dataset_name}: source does not exist")
                    link_results[dataset_name] = False
                    continue
                
                # Remove existing link if force is True
                if link_path.exists() or link_path.is_symlink():
                    if force:
                        if link_path.is_symlink():
                            link_path.unlink()
                        else:
                            shutil.rmtree(link_path)
                        print(f"Removed existing link: {link_path}")
                    else:
                        print(f"✓ Link already exists: {dataset_name}")
                        link_results[dataset_name] = True
                        continue
                
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
                    print(f"✓ Created link: {dataset_name} -> {source_path}")
                    link_results[dataset_name] = True
                else:
                    print(f"❌ Failed to create link: {dataset_name}")
                    link_results[dataset_name] = False
                    
            except Exception as e:
                print(f"❌ Error creating link for {dataset_name}: {e}")
                link_results[dataset_name] = False
        
        return link_results
    
    def check_dataset_integrity(self, dataset_name: str, sample_size: int = 100) -> Dict[str, any]:
        """Check dataset integrity by sampling files."""
        print(f"Checking integrity of {dataset_name} (sampling {sample_size} files)...")
        
        dataset_path = self.base_path / dataset_name
        
        integrity_result = {
            'dataset_name': dataset_name,
            'total_files': 0,
            'sampled_files': 0,
            'corrupted_files': [],
            'missing_files': [],
            'file_types': {},
            'status': 'unknown'
        }
        
        if not dataset_path.exists():
            integrity_result['status'] = 'missing'
            return integrity_result
        
        try:
            # Get all files
            all_files = list(dataset_path.rglob('*'))
            all_files = [f for f in all_files if f.is_file()]
            integrity_result['total_files'] = len(all_files)
            
            # Sample files for checking
            sample_files = random.sample(all_files, min(sample_size, len(all_files)))
            integrity_result['sampled_files'] = len(sample_files)
            
            # Check each sampled file
            for file_path in sample_files:
                try:
                    # Check if file is readable
                    with open(file_path, 'rb') as f:
                        # Read first few bytes to check if file is accessible
                        f.read(1024)
                    
                    # Count file types
                    file_ext = file_path.suffix.lower()
                    integrity_result['file_types'][file_ext] = integrity_result['file_types'].get(file_ext, 0) + 1
                    
                except Exception as e:
                    integrity_result['corrupted_files'].append({
                        'file': str(file_path.relative_to(dataset_path)),
                        'error': str(e)
                    })
            
            # Determine status
            if integrity_result['corrupted_files']:
                integrity_result['status'] = 'corrupted'
            else:
                integrity_result['status'] = 'healthy'
            
            print(f"✓ Integrity check complete: {len(integrity_result['corrupted_files'])} corrupted files found")
            
        except Exception as e:
            integrity_result['status'] = 'error'
            print(f"❌ Error during integrity check: {e}")
        
        return integrity_result
    
    def create_dataset_subset(self, source_dataset: str, target_dataset: str, 
                            sample_count: int, random_seed: int = 42) -> bool:
        """Create a subset of an existing dataset."""
        print(f"Creating {target_dataset} with {sample_count} samples from {source_dataset}...")
        
        source_path = self.base_path / source_dataset
        target_path = self.base_path / target_dataset
        
        if not source_path.exists():
            print(f"❌ Source dataset does not exist: {source_path}")
            return False
        
        if target_path.exists():
            print(f"⚠️  Target dataset already exists: {target_path}")
            response = input("Remove existing dataset? (y/n): ").strip().lower()
            if response == 'y':
                shutil.rmtree(target_path)
            else:
                return False
        
        try:
            # Set random seed for reproducibility
            random.seed(random_seed)
            
            # Create target directory structure
            target_path.mkdir(parents=True, exist_ok=True)
            
            # Get all files from source, organized by subdirectory
            source_files = {}
            for file_path in source_path.rglob('*'):
                if file_path.is_file():
                    # Get relative path from source
                    rel_path = file_path.relative_to(source_path)
                    subdir = rel_path.parts[0] if len(rel_path.parts) > 1 else ''
                    
                    if subdir not in source_files:
                        source_files[subdir] = []
                    source_files[subdir].append(file_path)
            
            # Sample files from each subdirectory proportionally
            copied_count = 0
            for subdir, files in source_files.items():
                if not files:
                    continue
                
                # Calculate sample size for this subdirectory
                subdir_sample_size = min(sample_count, len(files))
                if len(source_files) > 1:  # Multiple subdirectories
                    subdir_sample_size = min(sample_count // len(source_files), len(files))
                
                # Sample files
                sampled_files = random.sample(files, subdir_sample_size)
                
                # Copy files
                for source_file in sampled_files:
                    rel_path = source_file.relative_to(source_path)
                    target_file = target_path / rel_path
                    
                    # Create parent directories
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy file
                    shutil.copy2(source_file, target_file)
                    copied_count += 1
                    
                    if copied_count >= sample_count:
                        break
                
                if copied_count >= sample_count:
                    break
            
            print(f"✓ Created {target_dataset} with {copied_count} files")
            
            # Update dataset configuration
            self.dataset_configs[target_dataset] = {
                'description': f'{target_dataset} subset ({copied_count} samples)',
                'expected_size': f'~{self._estimate_size(copied_count)}',
                'min_files': copied_count,
                'subdirs': list(source_files.keys())
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating dataset subset: {e}")
            return False
    
    def get_dataset_statistics(self, dataset_name: str) -> Dict[str, any]:
        """Get detailed statistics for a dataset."""
        print(f"Analyzing dataset statistics: {dataset_name}")
        
        dataset_path = self.base_path / dataset_name
        
        stats = {
            'dataset_name': dataset_name,
            'path': str(dataset_path),
            'exists': dataset_path.exists(),
            'total_files': 0,
            'total_size_bytes': 0,
            'file_types': {},
            'subdirectories': {},
            'largest_files': [],
            'creation_time': None,
            'modification_time': None
        }
        
        if not dataset_path.exists():
            return stats
        
        try:
            # Get directory timestamps
            dir_stat = dataset_path.stat()
            stats['creation_time'] = dir_stat.st_ctime
            stats['modification_time'] = dir_stat.st_mtime
            
            # Analyze all files
            file_sizes = []
            
            for file_path in dataset_path.rglob('*'):
                if file_path.is_file():
                    file_stat = file_path.stat()
                    file_size = file_stat.st_size
                    
                    stats['total_files'] += 1
                    stats['total_size_bytes'] += file_size
                    
                    # Track file types
                    file_ext = file_path.suffix.lower()
                    if file_ext not in stats['file_types']:
                        stats['file_types'][file_ext] = {'count': 0, 'total_size': 0}
                    stats['file_types'][file_ext]['count'] += 1
                    stats['file_types'][file_ext]['total_size'] += file_size
                    
                    # Track subdirectories
                    rel_path = file_path.relative_to(dataset_path)
                    if len(rel_path.parts) > 1:
                        subdir = rel_path.parts[0]
                        if subdir not in stats['subdirectories']:
                            stats['subdirectories'][subdir] = {'count': 0, 'total_size': 0}
                        stats['subdirectories'][subdir]['count'] += 1
                        stats['subdirectories'][subdir]['total_size'] += file_size
                    
                    # Track largest files
                    file_sizes.append((file_size, str(file_path.relative_to(dataset_path))))
            
            # Get top 10 largest files
            file_sizes.sort(reverse=True)
            stats['largest_files'] = file_sizes[:10]
            
            print(f"✓ Statistics complete: {stats['total_files']} files, {self._format_size(stats['total_size_bytes'])}")
            
        except Exception as e:
            print(f"❌ Error analyzing statistics: {e}")
        
        return stats
    
    def generate_dataset_report(self, output_file: Optional[str] = None) -> Dict:
        """Generate comprehensive dataset report."""
        print("=== Generating Dataset Report ===\n")
        
        report = {
            'base_path': str(self.base_path),
            'project_data_dir': str(self.project_data_dir),
            'timestamp': self._get_timestamp(),
            'datasets': {},
            'summary': {}
        }
        
        # Validate all datasets
        validation_results = self.validate_all_datasets()
        
        # Get statistics for each dataset
        for dataset_name in self.dataset_configs.keys():
            print(f"\nAnalyzing {dataset_name}...")
            
            dataset_info = {
                'config': self.dataset_configs[dataset_name],
                'validation': validation_results.get(dataset_name, {}),
                'statistics': self.get_dataset_statistics(dataset_name),
                'integrity': None
            }
            
            # Check integrity for existing datasets
            if validation_results.get(dataset_name, {}).get('status') == 'valid':
                dataset_info['integrity'] = self.check_dataset_integrity(dataset_name, sample_size=50)
            
            report['datasets'][dataset_name] = dataset_info
        
        # Generate summary
        total_datasets = len(self.dataset_configs)
        valid_datasets = sum(1 for d in report['datasets'].values() 
                           if d['validation'].get('status') == 'valid')
        total_files = sum(d['statistics'].get('total_files', 0) 
                         for d in report['datasets'].values())
        total_size = sum(d['statistics'].get('total_size_bytes', 0) 
                        for d in report['datasets'].values())
        
        report['summary'] = {
            'total_datasets': total_datasets,
            'valid_datasets': valid_datasets,
            'total_files': total_files,
            'total_size_bytes': total_size,
            'total_size_formatted': self._format_size(total_size)
        }
        
        print(f"\n=== Report Summary ===")
        print(f"Valid datasets: {valid_datasets}/{total_datasets}")
        print(f"Total files: {total_files:,}")
        print(f"Total size: {self._format_size(total_size)}")
        
        # Save report if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\nReport saved to: {output_file}")
        
        return report
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size in bytes to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
    
    def _estimate_size(self, file_count: int) -> str:
        """Estimate dataset size based on file count."""
        # Rough estimate: average 1MB per file
        estimated_bytes = file_count * 1024 * 1024
        return self._format_size(estimated_bytes)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(description="Dataset management utilities")
    parser.add_argument("--base-path", default="/home/argus/Workspace/dataset",
                       help="Base dataset path")
    parser.add_argument("--validate", action="store_true",
                       help="Validate all datasets")
    parser.add_argument("--create-links", action="store_true",
                       help="Create symbolic links")
    parser.add_argument("--force-links", action="store_true",
                       help="Force recreate symbolic links")
    parser.add_argument("--integrity-check", metavar="DATASET",
                       help="Check integrity of specific dataset")
    parser.add_argument("--create-subset", nargs=3, metavar=("SOURCE", "TARGET", "COUNT"),
                       help="Create dataset subset: source target count")
    parser.add_argument("--statistics", metavar="DATASET",
                       help="Get statistics for specific dataset")
    parser.add_argument("--report", metavar="OUTPUT_FILE",
                       help="Generate comprehensive report")
    
    args = parser.parse_args()
    
    manager = DatasetManager(args.base_path)
    
    if args.validate:
        manager.validate_all_datasets()
    
    if args.create_links:
        manager.create_symbolic_links(force=args.force_links)
    
    if args.integrity_check:
        result = manager.check_dataset_integrity(args.integrity_check)
        print(f"Integrity check result: {result}")
    
    if args.create_subset:
        source, target, count = args.create_subset
        manager.create_dataset_subset(source, target, int(count))
    
    if args.statistics:
        stats = manager.get_dataset_statistics(args.statistics)
        print(f"Statistics: {json.dumps(stats, indent=2, default=str)}")
    
    if args.report:
        manager.generate_dataset_report(args.report)
    
    # If no specific action, run validation
    if not any([args.validate, args.create_links, args.integrity_check, 
                args.create_subset, args.statistics, args.report]):
        manager.validate_all_datasets()


if __name__ == "__main__":
    main()