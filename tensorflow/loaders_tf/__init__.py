"""
TensorFlow data loader factory for DewarpNet.
Equivalent to PyTorch loaders/__init__.py
"""

from .doc3dwc_loader_tf import create_wc_data_loaders, Doc3DWCDataset
from .doc3dbmnoimgc_loader_tf import create_bm_data_loaders, Doc3DBMDataset


def get_loader_tf(name: str):
    """
    Get data loader factory function by name.
    
    Args:
        name: Name of the loader ('doc3dwc' or 'doc3dbmnic')
        
    Returns:
        Data loader factory function
    """
    if name.lower() == 'doc3dwc':
        return create_wc_data_loaders
    elif name.lower() == 'doc3dbmnic':
        return create_bm_data_loaders
    else:
        raise ValueError(f"Unknown loader: {name}. Supported: 'doc3dwc', 'doc3dbmnic'")


def get_dataset_class_tf(name: str):
    """
    Get dataset class by name.
    
    Args:
        name: Name of the dataset ('doc3dwc' or 'doc3dbmnic')
        
    Returns:
        Dataset class
    """
    if name.lower() == 'doc3dwc':
        return Doc3DWCDataset
    elif name.lower() == 'doc3dbmnic':
        return Doc3DBMDataset
    else:
        raise ValueError(f"Unknown dataset: {name}. Supported: 'doc3dwc', 'doc3dbmnic'")


# Registry for easy access
LOADER_REGISTRY = {
    'doc3dwc': {
        'factory': create_wc_data_loaders,
        'dataset_class': Doc3DWCDataset,
        'description': 'World coordinate regression from RGB images'
    },
    'doc3dbmnic': {
        'factory': create_bm_data_loaders,
        'dataset_class': Doc3DBMDataset,
        'description': 'Backward mapping from world coordinates and albedo'
    }
}


if __name__ == "__main__":
    # Test loader factory
    print("Testing TensorFlow data loader factory...")
    
    # Test WC loader
    wc_loader_factory = get_loader_tf('doc3dwc')
    print(f"WC loader factory: {wc_loader_factory}")
    
    # Test BM loader
    bm_loader_factory = get_loader_tf('doc3dbmnic')
    print(f"BM loader factory: {bm_loader_factory}")
    
    # Test dataset classes
    wc_dataset_class = get_dataset_class_tf('doc3dwc')
    bm_dataset_class = get_dataset_class_tf('doc3dbmnic')
    print(f"WC dataset class: {wc_dataset_class}")
    print(f"BM dataset class: {bm_dataset_class}")
    
    print("✓ TensorFlow data loader factory test passed!")