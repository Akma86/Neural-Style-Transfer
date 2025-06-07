# 🖼️ Neural Style Transfer Research Assistant Project

## 📌 Overview
This project explores the implementation and analysis of **Neural Style Transfer (NST)**, a deep learning technique that merges the content of one image with the artistic style of another. The research is conducted as part of a faculty-supervised academic exploration, where I serve as a research assistant.

The goal is to investigate various NST techniques — from classical approaches introduced by Gatys et al. (2015) to modern real-time implementations using convolutional neural networks.

## 🧠 Objectives
- Understand and reproduce foundational NST algorithms.
- Analyze trade-offs between speed, quality, and flexibility across different NST models.
- Assist in documenting and experimenting with variations of NST for academic purposes.

## 🛠️ Methods
We implement and evaluate two main types of Neural Style Transfer:

1. **Optimization-based NST (Gatys et al.)**
   - Iteratively minimizes content and style loss using backpropagation.
   - Offers high-quality stylization but is computationally intensive.

2. **Fast Style Transfer (Johnson et al.)**
   - Uses a feed-forward network trained for a single or multiple styles.
   - Enables real-time inference with lower resource cost.

## 🧪 Technologies Used
- Python 3.x
- TensorFlow or PyTorch
- OpenCV & PIL
- Google Colab (for GPU acceleration)
- Matplotlib, NumPy


## 📊 Sample Results

| Content Image | Style Image | Stylized Output |
|---------------|-------------|-----------------|
| ![](images/content/city.jpg) | ![](images/style/starry_night.jpg) | ![](images/output/city_starry.jpg) |

## 👨‍🔬 Research Assistant Role
As a research assistant under the supervision of **Aditya Firmansyah**, my responsibilities include:
- Reproducing and validating key NST algorithms.
- Creating comparative analyses of NST variants.
- Exploring possible applications in educational or creative domains.

## 📌 Notes
This repository is a research companion and may evolve as experiments progress. For collaborations or academic inquiries, feel free to contact me.
