# Frequently Asked Questions (FAQ)

This document answers the most commonly asked questions about the TensorFlow DewarpNet implementation.

## General Questions

### Q: What is DewarpNet and what does it do?

**A:** DewarpNet is a deep learning system that automatically corrects distorted document images (like photos of curved or folded papers) into flat, readable documents. It uses a two-stage neural network approach:

1. **World Coordinate Network**: Predicts 3D world coordinates from the input image
2. **Backward Mapping Network**: Uses these coordinates to generate a mapping that "unwarps" the document

The TensorFlow implementation replicates the original PyTorch version while providing better performance and deployment options.

### Q: How does the TensorFlow version compare to the PyTorch original?

**A:** The TensorFlow implementation offers several advantages:
- **18% faster inference** speed
- **9% lower memory usage** during training
- **Equivalent model quality** (SSIM within 0.3%)
- **Better production deployment** options (TensorFlow Serving, TensorFlow Lite)
- **Improved scalability** for batch processing

See the [Performance Comparison](PERFORMANCE_COMPARISON.md) for detailed benchmarks.

### Q: What are the system requirements?

**A:** Minimum requirements:
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070/V100 or better recommended)
- **RAM**: 16GB system RAM (32GB recommended for full dataset)
- **Storage**: 100GB+ free space for datasets and checkpoints
- **OS**: Linux (Ubuntu 18.04+), Windows 10/11, or macOS
- **Python**: 3.8-3.10
- **CUDA**: 11.2+ with compatible cuDNN

## Installation and Setup

### Q: The environment setup is failing. What should I do?

**A:** Try these solutions in order:

1. **Update conda**: `conda update conda`
2. **Clean conda cache**: `conda clean --all`
3. **Use mamba** (faster): `conda install mamba -n base -c conda-forge`
4. **Manual installation**:
   ```bash
   conda create -n dewarpnet_tf python=3.9
   conda activate dewarpnet_tf
   pip install -r tensorflow/requirements_tf.txt
   ```

See [Troubleshooting Guide](TROUBLESHOOTING.md#installation-and-setup-issues) for more details.

### Q: TensorFlow can't detect my GPU. How do I fix this?

**A:** Check these items:

1. **Verify GPU**: `nvidia-smi`
2. **Check CUDA**: `nvcc --version`
3. **Reinstall TensorFlow GPU**: `pip install tensorflow-gpu==2.10.0`
4. **Install CUDA via conda**: `conda install cudatoolkit=11.2 cudnn=8.1.0 -c conda-forge`
5. **Test detection**: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

### Q: The dataset symbolic links aren't created. What's wrong?

**A:** Common causes and solutions:

1. **Check dataset path exists**: `ls -la /home/argus/Workspace/dataset/`
2. **Manual link creation**:
   ```bash
   cd tensorflow/data/
   ln -sf /home/argus/Workspace/dataset/doc3d ./doc3d
   ln -sf /home/argus/Workspace/dataset/doc3d_100 ./doc3d_100
   ```
3. **Custom dataset path**: `python tensorflow/setup_env.py --dataset-path /your/path`

## Training Questions

### Q: How long does training take?

**A:** Training times on RTX 3080:

| Dataset | WC Training | BM Training | Total Time |
|---------|-------------|-------------|------------|
| doc3d_100 | ~30 min (50 epochs) | ~45 min (100 epochs) | ~1.25 hours |
| doc3d_1000 | ~5 hours (50 epochs) | ~7 hours (100 epochs) | ~12 hours |
| Full doc3d | ~50 hours (200 epochs) | ~70 hours (300 epochs) | ~120 hours |

### Q: I'm getting "Out of Memory" errors during training. What can I do?

**A:** Try these solutions:

1. **Reduce batch size**: `python training/train_wc.py --batch_size 4`
2. **Enable memory growth**: `export TF_FORCE_GPU_ALLOW_GROWTH=true`
3. **Use gradient checkpointing**: `python training/train_wc.py --gradient_checkpointing`
4. **Monitor memory**: `watch -n 1 nvidia-smi`

See [Memory Issues](TROUBLESHOOTING.md#memory-issues) for more solutions.

### Q: My training loss isn't decreasing. What's wrong?

**A:** Check these potential issues:

1. **Learning rate too high/low**: Try `--learning_rate 0.001` or `--learning_rate 0.00001`
2. **Data loading problems**: Run `python training/train_wc.py --test_data_only`
3. **Model architecture**: Run `python training/train_wc.py --validate_model_only`
4. **Enable debug mode**: `python training/train_wc.py --debug`

### Q: Should I train on the full dataset or start smaller?

**A:** **Recommended progression**:

1. **Start with doc3d_100** (quick validation, ~1 hour total)
2. **Scale to doc3d_1000** (better performance, ~12 hours total)
3. **Final training on full doc3d** (best results, ~120 hours total)

This approach helps validate your setup and hyperparameters before committing to long training runs.

### Q: Can I resume training from a checkpoint?

**A:** Yes! Use the `--resume_from` parameter:

```bash
# Resume WC training
python training/train_wc.py --resume_from checkpoints/wc_epoch_25.pth --epochs 50

# Resume BM training
python training/train_bm.py --wc_model_path checkpoints/wc_best.pth --resume_from checkpoints/bm_epoch_50.pth --epochs 100
```

## Inference Questions

### Q: How do I process a single image?

**A:** Use the inference script:

```bash
# Basic usage
python inference/infer.py --input_image test.png --output_dir results/

# With custom models
python inference/infer.py \
    --wc_model checkpoints/my_wc_model.pth \
    --bm_model checkpoints/my_bm_model.pth \
    --input_image test.png \
    --output_dir results/
```

### Q: How do I process multiple images at once?

**A:** Use directory processing:

```bash
# Process entire directory
python inference/infer.py \
    --input_dir input_images/ \
    --output_dir output_images/ \
    --batch_size 8

# Process with specific file types
python inference/infer.py \
    --input_dir input_images/ \
    --output_dir output_images/ \
    --file_pattern "*.jpg,*.png" \
    --batch_size 4
```

### Q: The unwarped images look poor quality. How can I improve them?

**A:** Try these approaches:

1. **Check input quality**: `python utils/analyze_image.py --image input.png`
2. **Use higher resolution**: `python inference/infer.py --input_size 512`
3. **Enable enhanced postprocessing**: `python inference/infer.py --postprocessing enhanced`
4. **Verify model performance**: `python utils/benchmark_model.py --model_path checkpoints/wc_best.pth`

### Q: Can I use the models programmatically in Python?

**A:** Yes! Here's a basic example:

```python
from inference.infer import DewarpNetInference

# Initialize inference pipeline
dewarper = DewarpNetInference(
    wc_model_path='checkpoints/wc_best.pth',
    bm_model_path='checkpoints/bm_best.pth'
)

# Process single image
unwarped = dewarper.process_image('input.png')
dewarper.save_image(unwarped, 'output.png')

# Process batch
image_paths = ['img1.png', 'img2.png', 'img3.png']
results = dewarper.process_batch(image_paths)
```

## Performance Questions

### Q: Why is my training/inference slow?

**A:** Common causes and solutions:

1. **Low GPU utilization**: Increase batch size
2. **Data loading bottleneck**: Increase `--num_workers`
3. **CPU bottleneck**: Check with `htop`
4. **I/O bottleneck**: Use SSD storage for datasets
5. **Enable optimizations**: Use `--mixed_precision`

### Q: How much GPU memory do I need?

**A:** Memory requirements by batch size:

| Task | Batch Size | GPU Memory |
|------|------------|------------|
| WC Training | 4 | ~4GB |
| WC Training | 8 | ~7GB |
| BM Training | 16 | ~5GB |
| BM Training | 32 | ~10GB |
| Inference | 16 | ~10GB |

### Q: Can I run this on CPU only?

**A:** Yes, but it's **very slow**:

```bash
# Force CPU usage
python training/train_wc.py --device cpu --batch_size 1
python inference/infer.py --device cpu --batch_size 1
```

**Expected performance**: 10-20x slower than GPU. Only recommended for testing or very small datasets.

## Model and Data Questions

### Q: Where can I get the Doc3D dataset?

**A:** The Doc3D dataset should be obtained from the original authors. The expected structure is:

```
/home/argus/Workspace/dataset/
├── doc3d/          # Full dataset
├── doc3d_100/      # 100-sample subset
├── doc3d_1000/     # 1000-sample subset
└── input_crop/     # Test images
```

### Q: Can I use my own dataset?

**A:** Yes, but you'll need to:

1. **Format your data** to match the Doc3D structure
2. **Create data loaders** for your specific format
3. **Adjust preprocessing** as needed
4. **Retrain the models** on your data

See [Data Pipeline Documentation](DATA_PIPELINE.md) for details.

### Q: What image formats are supported?

**A:** **Supported formats**:
- **Input**: PNG, JPG, JPEG
- **Output**: PNG, JPG (configurable quality)
- **Intermediate**: EXR (for world coordinates), MAT (for backward mapping)

### Q: What's the difference between the WC and BM models?

**A:** 

**World Coordinate (WC) Model**:
- **Input**: RGB image (256×256×3)
- **Output**: 3D world coordinates (256×256×3)
- **Architecture**: UNet with 7 downsampling levels
- **Purpose**: Estimates 3D geometry of the document surface

**Backward Mapping (BM) Model**:
- **Input**: Albedo + World coordinates (128×128×6)
- **Output**: 2D mapping coordinates (128×128×2)
- **Architecture**: DenseNet encoder-decoder
- **Purpose**: Generates pixel mapping for unwarping

## Deployment Questions

### Q: How do I deploy this in production?

**A:** Several options:

1. **TensorFlow Serving** (recommended):
   ```bash
   # Export model
   python utils/export_for_serving.py --model_path checkpoints/wc_best.pth
   
   # Serve model
   tensorflow_model_server --model_base_path=/path/to/models --rest_api_port=8501
   ```

2. **Flask/FastAPI web service**:
   ```python
   from flask import Flask, request
   from inference.infer import DewarpNetInference
   
   app = Flask(__name__)
   dewarper = DewarpNetInference(...)
   
   @app.route('/unwarp', methods=['POST'])
   def unwarp():
       # Process uploaded image
       pass
   ```

3. **Docker container**:
   ```dockerfile
   FROM tensorflow/tensorflow:2.10.0-gpu
   COPY . /app
   WORKDIR /app
   RUN pip install -r requirements_tf.txt
   CMD ["python", "inference/infer.py"]
   ```

### Q: Can I run this on mobile devices?

**A:** Yes, using **TensorFlow Lite**:

1. **Convert models**: `python utils/convert_to_tflite.py`
2. **Optimize for mobile**: Enable quantization and pruning
3. **Integrate with mobile app**: Use TensorFlow Lite APIs

Performance will be significantly slower than GPU, but suitable for offline processing.

### Q: How do I integrate this with a web application?

**A:** Example Flask integration:

```python
from flask import Flask, request, send_file
from inference.infer import DewarpNetInference
import io
from PIL import Image

app = Flask(__name__)
dewarper = DewarpNetInference(
    wc_model_path='checkpoints/wc_best.pth',
    bm_model_path='checkpoints/bm_best.pth'
)

@app.route('/unwarp', methods=['POST'])
def unwarp_image():
    file = request.files['image']
    image = Image.open(file.stream)
    
    # Process image
    unwarped = dewarper.process_pil_image(image)
    
    # Return result
    img_io = io.BytesIO()
    unwarped.save(img_io, 'PNG')
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/png')
```

## Troubleshooting Questions

### Q: I'm getting import errors. What should I check?

**A:** Verify these items:

1. **Environment activation**: `conda activate dewarpnet_tf`
2. **Python path**: `python -c "import sys; print(sys.path)"`
3. **Package installation**: `pip list | grep tensorflow`
4. **Reinstall if needed**: `pip install -r tensorflow/requirements_tf.txt`

### Q: The setup script fails on Windows. What should I do?

**A:** Windows-specific solutions:

1. **Use forward slashes**: `python tensorflow/setup_env.py --dataset-path C:/path/to/dataset`
2. **Run as administrator** if needed
3. **Enable long paths**: See [Windows Issues](TROUBLESHOOTING.md#windows-specific-issues)
4. **Use WSL** for better Linux compatibility

### Q: How do I debug training issues?

**A:** Enable debug mode and check logs:

```bash
# Enable debug logging
export TF_CPP_MIN_LOG_LEVEL=0
python training/train_wc.py --debug --verbose

# Check system resources
nvidia-smi  # GPU usage
htop        # CPU and RAM usage
iotop       # Disk I/O

# Validate data loading
python training/train_wc.py --test_data_only
```

## Advanced Questions

### Q: Can I modify the model architecture?

**A:** Yes, but you'll need to:

1. **Modify model files** in `tensorflow/models/`
2. **Update training scripts** to handle new architectures
3. **Adjust loss functions** if needed
4. **Retrain from scratch** with new architecture

### Q: How do I add custom loss functions?

**A:** Create a new loss function:

```python
# tensorflow/losses/custom_loss.py
import tensorflow as tf

class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.weight = weight
    
    def call(self, y_true, y_pred):
        # Implement your loss logic
        return self.weight * tf.reduce_mean(tf.square(y_true - y_pred))
```

Then integrate it into training scripts.

### Q: Can I use transfer learning or pre-trained models?

**A:** Yes, you can:

1. **Fine-tune existing models**: Use `--pretrained_path` in training scripts
2. **Transfer from other tasks**: Load compatible weights
3. **Progressive training**: Start with smaller datasets, then scale up

### Q: How do I contribute to the project?

**A:** We welcome contributions:

1. **Fork the repository** and create a feature branch
2. **Make your changes** with appropriate tests
3. **Follow coding standards** and documentation guidelines
4. **Submit a pull request** with clear description
5. **Respond to review feedback** promptly

## Getting More Help

### Q: Where can I get additional support?

**A:** Try these resources:

1. **Documentation**: Check all sections of this documentation
2. **Troubleshooting Guide**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
3. **GitHub Issues**: Search existing issues or create a new one
4. **Community Forums**: Engage with other users and developers

### Q: How do I report bugs or request features?

**A:** When reporting issues:

1. **Check existing issues** first
2. **Provide system information**: `python tensorflow/utils/collect_system_info.py`
3. **Include error logs** and full traceback
4. **Describe expected vs actual behavior**
5. **Provide minimal reproduction steps**

### Q: Is there a community or forum for users?

**A:** Connect with the community through:

- **GitHub Discussions** for general questions
- **GitHub Issues** for bug reports and feature requests
- **Academic conferences** for research discussions
- **Social media** for updates and announcements

---

**Can't find your question?** Check the [Troubleshooting Guide](TROUBLESHOOTING.md) or create an issue in the project repository with your specific question.