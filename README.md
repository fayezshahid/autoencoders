# Face Recognition Autoencoders

A comprehensive implementation and evaluation of three autoencoder architectures for face recognition: Convolutional Autoencoder (CAE), Variational Autoencoder (VAE), and Adversarial Autoencoder (AAE).

## Architecture

### Convolutional Autoencoder (CAE)
- **Encoder**: 4 Conv2D layers (128→256→512→1024 filters) with BatchNorm and ReLU
- **Decoder**: 4 Conv2DTranspose layers with symmetric architecture
- **Loss**: Mean Squared Error (MSE)

### Variational Autoencoder (VAE)
- **Encoder**: Same as CAE + latent mean and variance layers
- **Decoder**: Symmetric to encoder
- **Loss**: Reconstruction loss + KL divergence

### Adversarial Autoencoder (AAE)
- **Encoder/Decoder**: Same as CAE
- **Discriminator**: 3-layer MLP (512→512→512→1)
- **Loss**: Reconstruction loss + Adversarial loss

## Key Features
- Custom data generator with train/test split
- Batch training with validation monitoring
- Early stopping and model checkpointing
- Comprehensive evaluation metrics

## Results

### Clustering Performance
- **Best Silhouette Scores**: CAE (0.XX), VAE (0.XX), AAE (0.XX)
- **Separation Scores**: Higher intra-class vs inter-class similarity
- **Multiple clustering algorithms**: K-Means, Spectral, GMM, Agglomerative

### Evaluation Metrics
- t-SNE visualization of learned embeddings
- Cosine similarity matrices
- Reconstruction quality (MSE, PSNR, SSIM)
- Adjusted Rand Index and Normalized Mutual Information

## Dataset & Models
Access the complete dataset and trained models: [Google Drive](https://drive.google.com/drive/folders/1jmdH-T0Hd1hPILRGIt3QDq5QVjWJDkTk?usp=sharing)

## Usage

### Training
```bash
python train.py --data_dir casia-webface --epochs 50 --batch_size 512 --models cae,vae,aae
```

### Evaluation
Run the `evaluate.ipynb` notebook for comprehensive model evaluation and visualization.

## Requirements
- TensorFlow/Keras
- NumPy, PIL, scikit-learn
- Matplotlib, seaborn
- FPDF for report generation