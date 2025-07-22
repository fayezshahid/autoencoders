#!/usr/bin/env python
"""
Complete training script for CNN Autoencoder (CAE), Adversarial Autoencoder (AAE), 
and Variational Autoencoder (VAE) on face recognition dataset.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Layer
from PIL import Image
import pickle
import argparse
import sys

# Set up GPU memory growth to avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class ImageDataGenerator(tf.keras.utils.Sequence):
    """Custom data generator for loading images in batches with train/test split."""
    
    def __init__(self, data_dir, batch_size, img_size, num_channels, file_ext, 
                 train_split=0.8, is_training=True, shuffle=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_channels = num_channels
        self.file_ext = file_ext
        self.shuffle = shuffle
        self.is_training = is_training
        self.train_split = train_split
        
        # Get all image files
        all_files = [f for f in os.listdir(self.data_dir) if f.endswith(self.file_ext)]
        all_files.sort()  # Sort for consistent splitting across runs
        
        # Split into train and test
        split_idx = int(len(all_files) * train_split)
        
        if is_training:
            self.file_names = all_files[:split_idx]
            split_type = "training"
        else:
            self.file_names = all_files[split_idx:]
            split_type = "validation"
        
        self.indexes = np.arange(len(self.file_names))
        if self.shuffle:
            np.random.shuffle(self.indexes)
        
        print(f"Found {len(self.file_names)} images in {data_dir} for {split_type}")
        print(f"Split ratio: {train_split:.1%} train, {1-train_split:.1%} validation")

    def __len__(self):
        return len(self.file_names) // self.batch_size

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_data = np.zeros((len(batch_indexes), self.img_size[0], self.img_size[1], self.num_channels), dtype=np.float32)
        
        for i, batch_idx in enumerate(batch_indexes):
            try:
                file_name = self.file_names[batch_idx]
                img = Image.open(os.path.join(self.data_dir, file_name))
                img = img.resize(self.img_size)
                img = np.array(img, dtype=np.float32) / 255.0
                batch_data[i] = img
            except Exception as e:
                print(f"Error loading image {file_name}: {e}")
                # Fill with zeros if image can't be loaded
                batch_data[i] = np.zeros((self.img_size[0], self.img_size[1], self.num_channels))
        
        return batch_data, batch_data

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def create_encoder(zdim=64, input_shape=(64, 64, 3)):
    """Create encoder network."""
    return tf.keras.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Conv2D(filters=128, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(filters=256, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(filters=512, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(filters=1024, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Flatten(),
        layers.Dense(units=zdim)
    ])

# def create_encoder_with_pretrained_resnet50(zdim=128, input_shape=(64, 64, 3)):
#     base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
#     freeze_layers = min(len(base_model.layers) - 10, int(len(base_model.layers) * 0.8))
#     for i, layer in enumerate(base_model.layers):
#         if i < freeze_layers:
#             layer.trainable = False
#         else:
#             layer.trainable = True
    
#     print(f"Frozen {freeze_layers} layers, training {len(base_model.layers) - freeze_layers} layers")
    
#     # Add custom head for face recognition
#     x = base_model.output
#     x = layers.GlobalAveragePooling2D()(x)
#     x = layers.Dense(512, activation='relu', name='fc1')(x)
#     x = layers.BatchNormalization(name='bn1')(x)
#     x = layers.Dropout(0.5, name='dropout1')(x)
#     x = layers.Dense(256, activation='relu', name='fc2')(x)
#     x = layers.BatchNormalization(name='bn2')(x)
#     x = layers.Dropout(0.3, name='dropout2')(x)
#     x = layers.Dense(zdim, name='embedding')(x)
    
#     encoder = Model(base_model.input, x, name=f'encoder_resnet50')
    
#     # Print model summary
#     print(f"\nEncoder architecture (resnet50):")
#     print(f"Input shape: {input_shape}")
#     print(f"Output shape: {encoder.output_shape}")

#     # Display the complete model summary
#     encoder.summary()

#     return encoder

def create_decoder(zdim=64, output_shape=(64, 64, 3)):
    """Create decoder network."""
    return tf.keras.Sequential([
        layers.InputLayer(input_shape=(zdim,)),
        layers.Dense(units=1024*4*4),
        layers.Reshape((4, 4, 1024)),
        layers.Conv2DTranspose(filters=512, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2DTranspose(filters=output_shape[2], kernel_size=4, strides=2, padding='same', activation='sigmoid')
    ])

def create_discriminator(zdim=64):
    """Create discriminator network for AAE."""
    return tf.keras.Sequential([
        layers.InputLayer(input_shape=(zdim,)),
        layers.Dense(units=512, activation='relu'),
        layers.Dense(units=512, activation='relu'),
        layers.Dense(units=512, activation='relu'),
        layers.Dense(units=1)
    ])

def train_cae(train_gen, val_gen, epochs, zdim, save_dir):
    """Train Convolutional Autoencoder (CAE)."""
    print("Training CNN Autoencoder (CAE)...")
    
    encoder = create_encoder(zdim)
    decoder = create_decoder(zdim)
    
    autoencoder = tf.keras.Sequential([encoder, decoder])
    autoencoder.compile(optimizer=Adam(learning_rate=0.0002), loss='mse')
    
    # Setup callbacks
    checkpoint_path = os.path.join(save_dir, "cae_model_checkpoint_{epoch:02d}.h5")
    checkpoint = ModelCheckpoint(checkpoint_path, save_freq='epoch', save_best_only=False)
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    
    # Train the model
    history = autoencoder.fit(
        train_gen,
        epochs=epochs,
        steps_per_epoch=len(train_gen),
        validation_data=val_gen,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )
    
    # Save final models
    autoencoder.save(os.path.join(save_dir, "cae_autoencoder_final.h5"))
    encoder.save(os.path.join(save_dir, "cae_encoder_final.h5"))
    decoder.save(os.path.join(save_dir, "cae_decoder_final.h5"))
    
    # Save as pickle for compatibility
    with open(os.path.join(save_dir, 'cae_encoder.pkl'), 'wb') as f:
        pickle.dump(encoder, f)
    
    print("CAE training completed!")
    return autoencoder, encoder, decoder

def train_vae(train_gen, val_gen, epochs, zdim, save_dir):
    """Train Variational Autoencoder (VAE)."""
    print("Training Variational Autoencoder (VAE)...")

    class VAELossLayer(Layer):
        def call(self, inputs):
            encoder_inputs, outputs, z_mean, z_log_var = inputs

            reconstruction_loss = tf.reduce_sum(tf.square(encoder_inputs - outputs), axis=[1, 2, 3])
            kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

            self.add_loss(total_loss)
            return outputs
    
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], zdim), mean=0., stddev=1.)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
    # Encoder
    encoder_inputs = layers.Input(shape=(64, 64, 3))
    x = layers.Conv2D(filters=128, kernel_size=4, strides=2, padding='same', use_bias=False)(encoder_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=256, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=512, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=1024, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Flatten()(x)
    
    z_mean = layers.Dense(units=zdim, name='z_mean')(x)
    z_log_var = layers.Dense(units=zdim, name='z_log_var')(x)
    z = layers.Lambda(sampling, output_shape=(zdim,), name='z')([z_mean, z_log_var])
    
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
    
    # Decoder
    latent_inputs = layers.Input(shape=(zdim,), name='z_sampling')
    x = layers.Dense(units=1024*4*4)(latent_inputs)
    x = layers.Reshape((4, 4, 1024))(x)
    x = layers.Conv2DTranspose(filters=512, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    decoder_outputs = layers.Conv2DTranspose(filters=3, kernel_size=4, strides=2, padding='same', activation='sigmoid')(x)
    
    decoder = Model(latent_inputs, decoder_outputs, name='decoder')
    
    # VAE model with loss
    outputs = decoder(encoder(encoder_inputs)[2])
    outputs_with_loss = VAELossLayer(name='vae_loss')(
        [encoder_inputs, outputs, z_mean, z_log_var]
    )
    vae = Model(encoder_inputs, outputs_with_loss, name='vae')
    
    # VAE loss
    # reconstruction_loss = tf.keras.losses.mse(K.flatten(encoder_inputs), K.flatten(outputs))
    # reconstruction_loss *= 64 * 64 * 3
    # kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    # kl_loss = K.sum(kl_loss, axis=-1)
    # kl_loss *= -0.5
    # vae_loss = K.mean(reconstruction_loss + kl_loss)
    # vae.add_loss(vae_loss)
    
    vae.compile(optimizer=Adam(learning_rate=0.0002))
    
    # Setup callbacks
    checkpoint_path = os.path.join(save_dir, "vae_model_checkpoint_{epoch:02d}.h5")
    checkpoint = ModelCheckpoint(checkpoint_path, save_freq='epoch', save_best_only=False)
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    
    # Train the model
    history = vae.fit(
        train_gen,
        epochs=epochs,
        steps_per_epoch=len(train_gen),
        validation_data=val_gen,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )
    
    # Save final models
    vae.save(os.path.join(save_dir, "vae_model_final.h5"))
    encoder.save(os.path.join(save_dir, "vae_encoder_final.h5"))
    decoder.save(os.path.join(save_dir, "vae_decoder_final.h5"))
    
    # Save as pickle for compatibility
    with open(os.path.join(save_dir, 'vae_encoder.pkl'), 'wb') as f:
        pickle.dump(encoder, f)
    
    print("VAE training completed!")
    return vae, encoder, decoder

def train_aae(train_gen, val_gen, epochs, zdim, save_dir):
    """Train Adversarial Autoencoder (AAE) with validation monitoring."""
    print("Training Adversarial Autoencoder (AAE)...")
    
    encoder = create_encoder(zdim)
    decoder = create_decoder(zdim)
    discriminator = create_discriminator(zdim)
    
    # Loss functions
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    mse_loss = tf.keras.losses.MeanSquaredError()
    
    # Optimizers
    gen_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    disc_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    
    # Training loop with validation
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_disc_loss = 0
        epoch_gen_loss = 0
        
        # Training phase
        for i, (x_real, _) in enumerate(train_gen):
            batch_size = x_real.shape[0]
            
            # Train discriminator
            with tf.GradientTape() as tape:
                z_real = encoder(x_real, training=True)
                z_fake = tf.random.normal(shape=(batch_size, zdim))
                
                d_real = discriminator(z_real, training=True)
                d_fake = discriminator(z_fake, training=True)
                
                disc_loss_real = cross_entropy(tf.ones_like(d_real), d_real)
                disc_loss_fake = cross_entropy(tf.zeros_like(d_fake), d_fake)
                disc_loss = disc_loss_real + disc_loss_fake
                
            disc_grads = tape.gradient(disc_loss, discriminator.trainable_variables)
            disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
            
            # Train generator (encoder + decoder)
            with tf.GradientTape() as tape:
                z_encoded = encoder(x_real, training=True)
                x_reconstructed = decoder(z_encoded, training=True)
                
                d_encoded = discriminator(z_encoded, training=True)
                
                reconstruction_loss = mse_loss(x_real, x_reconstructed)
                adversarial_loss = cross_entropy(tf.ones_like(d_encoded), d_encoded)
                gen_loss = reconstruction_loss + 0.1 * adversarial_loss
                
            gen_grads = tape.gradient(gen_loss, encoder.trainable_variables + decoder.trainable_variables)
            gen_optimizer.apply_gradients(zip(gen_grads, encoder.trainable_variables + decoder.trainable_variables))
            
            # Convert to scalar values for accumulation
            epoch_disc_loss += disc_loss.numpy()
            epoch_gen_loss += gen_loss.numpy()
            
            if (i + 1) % 50 == 0:
                print(f"Batch {i+1}: Disc loss = {disc_loss.numpy():.4f}, Gen loss = {gen_loss.numpy():.4f}")
        
        # Validation phase
        val_gen_loss = 0
        val_disc_loss = 0
        
        for x_val, _ in val_gen:
            batch_size = x_val.shape[0]
            
            # Validation discriminator loss
            z_real = encoder(x_val, training=False)
            z_fake = tf.random.normal(shape=(batch_size, zdim))
            
            d_real = discriminator(z_real, training=False)
            d_fake = discriminator(z_fake, training=False)
            
            disc_loss_real = cross_entropy(tf.ones_like(d_real), d_real)
            disc_loss_fake = cross_entropy(tf.zeros_like(d_fake), d_fake)
            val_disc_loss += (disc_loss_real + disc_loss_fake).numpy()
            
            # Validation generator loss
            z_encoded = encoder(x_val, training=False)
            x_reconstructed = decoder(z_encoded, training=False)
            
            d_encoded = discriminator(z_encoded, training=False)
            
            reconstruction_loss = mse_loss(x_val, x_reconstructed)
            adversarial_loss = cross_entropy(tf.ones_like(d_encoded), d_encoded)
            val_gen_loss += (reconstruction_loss + 0.1 * adversarial_loss).numpy()
        
        # Calculate average losses
        avg_train_disc_loss = epoch_disc_loss / len(train_gen)
        avg_train_gen_loss = epoch_gen_loss / len(train_gen)
        avg_val_disc_loss = val_disc_loss / len(val_gen)
        avg_val_gen_loss = val_gen_loss / len(val_gen)
        
        print(f"Epoch {epoch+1} - Train: Disc={avg_train_disc_loss:.4f}, Gen={avg_train_gen_loss:.4f}")
        print(f"Epoch {epoch+1} - Val:   Disc={avg_val_disc_loss:.4f}, Gen={avg_val_gen_loss:.4f}")
        
        # Early stopping based on validation generator loss
        if avg_val_gen_loss < best_val_loss:
            best_val_loss = avg_val_gen_loss
            patience_counter = 0
            # Save best model
            encoder.save(os.path.join(save_dir, "aae_encoder_best.h5"))
            decoder.save(os.path.join(save_dir, "aae_decoder_best.h5"))
            discriminator.save(os.path.join(save_dir, "aae_discriminator_best.h5"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Save checkpoint every epoch
        encoder.save(os.path.join(save_dir, f"aae_encoder_epoch_{epoch+1}.h5"))
        decoder.save(os.path.join(save_dir, f"aae_decoder_epoch_{epoch+1}.h5"))
    
    # Save final models
    encoder.save(os.path.join(save_dir, "aae_encoder_final.h5"))
    decoder.save(os.path.join(save_dir, "aae_decoder_final.h5"))
    discriminator.save(os.path.join(save_dir, "aae_discriminator_final.h5"))
    
    # Save as pickle for compatibility
    with open(os.path.join(save_dir, 'aae_encoder.pkl'), 'wb') as f:
        pickle.dump(encoder, f)
    
    print("AAE training completed!")
    return encoder, decoder, discriminator

def main():
    parser = argparse.ArgumentParser(description='Train face recognition autoencoders')
    parser.add_argument('--data_dir', type=str, default='casia-webface', 
                        help='Directory containing training images')
    parser.add_argument('--save_dir', type=str, default='./models', 
                        help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=5, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512, 
                        help='Batch size for training')
    parser.add_argument('--zdim', type=int, default=128, 
                        help='Latent dimension size')
    parser.add_argument('--models', type=str, default='cae,aae,vae',
                        help='Comma-separated list of models to train (cae,aae,vae)')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup data generator
    img_size = (64, 64)
    num_channels = 3
    file_ext = ".jpg"
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist!")
        sys.exit(1)
    
    train_gen = ImageDataGenerator(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=img_size,
        num_channels=num_channels,
        file_ext=file_ext,
        train_split=0.8,
        is_training=True,
        shuffle=True
    )

    val_gen = ImageDataGenerator(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=img_size,
        num_channels=num_channels,
        file_ext=file_ext,
        train_split=0.8,
        is_training=False,
        shuffle=False
    )
    
    print(f"Training with {len(train_gen)} batches per epoch")
    
    # Parse models to train
    models_to_train = [m.strip().lower() for m in args.models.split(',')]
    
    # Train models
    if 'cae' in models_to_train:
        cae_autoencoder, cae_encoder, cae_decoder = train_cae(
            train_gen, val_gen, args.epochs, args.zdim, args.save_dir
        )
    
    if 'vae' in models_to_train:
        vae_model, vae_encoder, vae_decoder = train_vae(
            train_gen, val_gen, args.epochs, args.zdim, args.save_dir
        )

    if 'aae' in models_to_train:
        aae_encoder, aae_decoder, aae_discriminator = train_aae(
            train_gen, val_gen, args.epochs, args.zdim, args.save_dir
        )
    
    print("All training completed!")
    print(f"Models saved in: {args.save_dir}")
    
    # Print summary of saved files
    print("\nSaved files:")
    for file in os.listdir(args.save_dir):
        print(f"  - {file}")

if __name__ == "__main__":
    main()