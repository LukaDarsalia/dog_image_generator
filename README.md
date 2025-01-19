
# Dog Breed Image Generation with Conditional VAE

## Overview
This project explores generative modeling using a Conditional Variational Autoencoder (C-VAE) to generate dog images conditioned on their breed. It serves as an educational journey into conditional generative models, debugging, and architectural experiments.


## How It Works

### Architecture
1. **Encoder:** Encodes input into latent space with mean (`mu`) and log-variance (`logvar`).
2. **Reparameterization:** Samples latent vector `z = mu + epsilon * sigma`.
3. **Conditioning:** Incorporates breed-specific embeddings into `z`.
4. **Decoder:** Reconstructs breed-specific images from `z`.


### Loss Function
1. **Reconstruction Loss:** Measures the similarity between input and generated images:
   ```
   L_recon = (1 / N) * sum(||x_i - x_hat_i||^2)
   ```
2. **KL Divergence Loss:** Regularizes the latent space:
   ```
   L_KL = -0.5 * (1 / N) * sum(1 + log(sigma^2) - mu^2 - sigma^2)
   ```
3. **Total Loss:**
   ```
   L = L_recon + beta * L_KL
   ```

## Data
The dataset used for this project can be downloaded from the Kaggle notebook:
[Generative Dog Images](https://www.kaggle.com/code/lukadarsalia/generative-dog-images).  
The data is also available via the competition hosted on Kaggle.


## Results
Generated images show some intuition (e.g., noses, colors) but lack structural quality due to dataset and model constraints.


## Lessons Learned
1. Built my first Conditional VAE for vision tasks and learned its theory.
2. Explored debugging techniques like gradient flow analysis.
3. Understood challenges in generative modeling, balancing architecture and data quality.


## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/LukaDarsalia/dog_image_generator.git
   cd dog_image_generator
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
