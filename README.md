
# Practical No. 04: Image Generation using Deep Convolutional GAN (DCGAN)

**Student:** Soham Sabane  
**PRN:** 202401110070  
**Dataset:** Oxford Flowers 102  

## 1. Project Overview

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) using TensorFlow and Keras. The main objective is to generate synthetic flower images using a trained deep learning model.

DCGAN consists of two neural networks: a Generator and a Discriminator. The Generator creates synthetic flower images from random noise, while the Discriminator distinguishes real flower images from generated images. Both networks are trained adversarially to improve their performance.

The project evaluates the model using Generator and Discriminator loss curves, generated image grids, discriminator accuracy, real-versus-generated image comparisons, latent-space interpolation, and an optional Fréchet Inception Distance (FID) calculation.

## 2. Dataset Description

The project uses the **Oxford Flowers 102** dataset through TensorFlow Datasets.

- **Dataset name:** Oxford Flowers 102
- **Dataset source:** TensorFlow Datasets (`oxford_flowers102`)
- **Content:** Flower photographs belonging to 102 categories.
- **Data splits:** Training and validation datasets.
- **Images used:** Up to 2,000 images are selected for training in the notebook.

### Data Preprocessing

The images undergo the following preprocessing steps:

1. Convert images to RGB format.
2. Centre-crop images to a square.
3. Resize images to 64 × 64 pixels.
4. Convert pixel values to floating-point numbers.
5. Normalize pixel values from the range [0, 255] to [-1, 1].

Normalization is necessary because the Generator uses a `tanh` activation function, which produces output values in the range [-1, 1].

The dataset is downloaded automatically through TensorFlow Datasets when the notebook is executed for the first time. Internet access may be required.

## 3. DCGAN Architecture

The model contains two main components.

### 3.1 Generator

The Generator creates synthetic flower images from random latent vectors.

Main operations:
- Takes a 100-dimensional random noise vector as input.
- Uses a dense layer to create an initial feature representation.
- Reshapes the features into a multi-dimensional tensor.
- Uses transposed convolution layers to increase spatial dimensions.
- Applies batch normalization and LeakyReLU activation functions.
- Produces a 64 × 64 RGB image using a `tanh` output activation.

The Generator learns to create realistic-looking images that can fool the Discriminator.

### 3.2 Discriminator

The Discriminator determines whether an image is real or generated.

Main operations:
- Takes a 64 × 64 RGB image as input.
- Uses convolutional layers to extract image features.
- Uses LeakyReLU activation functions.
- Applies dropout to help reduce overfitting.
- Produces a real/fake prediction.

The Discriminator learns to identify differences between genuine flower photographs and synthetic images produced by the Generator.

## 4. Model Training

The Generator and Discriminator are trained together using adversarial learning.

During training:

1. Random noise vectors are supplied to the Generator.
2. The Generator creates synthetic flower images.
3. The Discriminator receives real and generated images.
4. The Discriminator learns to distinguish real images from synthetic images.
5. The Generator learns to produce images that are more likely to be classified as real.
6. The process is repeated over multiple training epochs.

### Training Configuration

- Latent dimension: 100
- Batch size: 64
- Number of epochs: 5
- Optimizer: Adam
- Learning rate: 0.0002
- Adam beta 1: 0.5
- Real-label smoothing: 0.9
- Data augmentation: Random horizontal and vertical flips

The notebook uses a limited number of images and epochs to demonstrate the training process. Longer training may improve image quality, depending on model stability and available computing resources.

## 5. Evaluation of the Model

The performance of the DCGAN is evaluated using the following methods.

### 5.1 Generator Loss

Generator loss measures how well the Generator fools the Discriminator. The Generator attempts to reduce this loss by producing images that the Discriminator classifies as real.

### 5.2 Discriminator Loss

Discriminator loss measures the Discriminator's ability to distinguish real images from generated images.

GAN losses may fluctuate during training because the Generator and Discriminator continuously compete with each other. Therefore, both loss curves must be analyzed together.

### 5.3 Loss Curves

The notebook plots Generator and Discriminator losses across training epochs.

These curves help observe training behaviour, stability, and changes in the competition between the two networks.

A loss curve alone cannot prove that the generated images are realistic. It must be considered alongside visual results.

### 5.4 Generated Image Grids

The trained Generator creates synthetic flower images from random noise.

The generated images can be inspected to evaluate:
- Image clarity
- Flower-like structures
- Colour variation
- Diversity among generated samples
- Improvement across training epochs

### 5.5 Real-versus-Generated Comparison

Real dataset images and generated images are displayed together to compare their visual characteristics.

This comparison helps identify whether the Generator has learned useful patterns from the training dataset.

### 5.6 Latent-Space Interpolation

Latent-space interpolation gradually changes the random input vector between two points. The resulting images demonstrate how the Generator's output changes as its input changes.

### 5.7 Fréchet Inception Distance (FID)

The notebook includes an optional FID calculation using features extracted by a pretrained InceptionV3 network.

FID compares the feature distributions of real and generated images. A lower FID generally indicates greater similarity between these distributions.

The FID result depends on the generated samples and evaluation setup. The optional calculation may require additional processing time and pretrained model weights.

## 6. Technologies Used

- Python
- TensorFlow
- Keras
- TensorFlow Datasets
- NumPy
- Matplotlib
- Pillow
- SciPy

These technologies are used for dataset processing, deep learning model development, training, visualization, and optional evaluation.

## 7. How to Run the Project

1. Open `DCGAN_Generative-AI.ipynb` in Google Colab or Jupyter Notebook.
2. Select a GPU runtime if available.
3. Execute the setup and library import cells.
4. Load the Oxford Flowers 102 dataset.
5. Run the image preprocessing steps.
6. Build the Generator and Discriminator.
7. Execute the training cells.
8. View the Generator and Discriminator loss curves.
9. Inspect the generated image grids and real-versus-generated comparison.
10. Run the optional FID evaluation if the required dependencies and computing resources are available.

If TensorFlow Datasets is not installed, run:

```python
!pip install tensorflow-datasets
```

The notebook saves generated image grids and Generator checkpoints in the `outputs/` directory.

## 8. Results and Discussion

The project demonstrates how a DCGAN can learn visual patterns from flower photographs and use them to generate synthetic images.

The Generator and Discriminator loss curves help analyze adversarial training behaviour, while the generated image grids provide a visual assessment of image quality and diversity.

Because the notebook is configured to use up to 2,000 images and five epochs, the generated images may have limited detail or contain visual artefacts. Results should be discussed using the actual images and loss values obtained after running the notebook.

The optional FID score provides an additional quantitative measure for comparing real and generated image distributions.

## 9. Conclusion

This project implements a Deep Convolutional Generative Adversarial Network for synthetic image generation using the Oxford Flowers 102 dataset.

The Generator learns to produce flower-like images from random noise, while the Discriminator learns to distinguish real photographs from generated samples. The model is evaluated through Generator and Discriminator loss curves, generated image grids, image comparisons, latent-space interpolation, and optional FID evaluation.

The practical demonstrates the basic working principles of Generative AI and adversarial learning. Further improvements may be achieved by increasing the number of training epochs, using more training images, and tuning the model architecture and hyperparameters.
