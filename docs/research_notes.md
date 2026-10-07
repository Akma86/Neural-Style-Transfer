# 🔬 Neural Style Transfer — Research Notes & Roadmap

This document compiles the research inquiries, theoretical explorations, and roadmap items investigated in this project under the supervision of **Aditya Firmansyah**.

---

## 📌 Research Questions & Technical Feasibility

### 1. Multi-Backbone & Hybrid Ensembles for Content and Style
* **Question**: *Can we combine different backbone architectures (e.g., EfficientNet for content extraction + VGG-19 for style representation, or ensembling two models per stream)?*
* **Analysis**: Yes. In traditional NST (Gatys et al., 2015), VGG-19 is favored due to its simple, unresidual convolutions that produce rich Gram matrix representations. However, modern backbones (EfficientNet, ConvNeXt, LeViT) offer distinct inductive biases:
  - **Content extraction**: Benefits from architectures with high semantic abstraction (EfficientNet-B0/V2, ConvNeXt).
  - **Style extraction**: Requires localized texture correlation. Standard residual connections can dilute Gram matrix stability unless specific intermediate layers are isolated.
* **Implementation in Repo**: See [`notebooks/fusion/01_fusion_model_nst.ipynb`](../notebooks/fusion/01_fusion_model_nst.ipynb), where hybrid optimization is formulated.

---

### 2. Fast Feedforward Neural Style Transfer (Deep Learning Inference)
* **Question**: *Can we replace per-image iterative optimization with end-to-end feedforward neural networks?*
* **Analysis**: Yes. While Gatys' optimization takes hundreds of gradient steps per image pair, feedforward networks (Johnson et al., 2016; Ulyanov et al., 2016; Huang & Belongie AdaIN, 2017) train an image transformation network with perceptual loss to achieve real-time (sub-second) stylization.
* **Implementation in Repo**: Explored in [`notebooks/tensorflow/05_deep_learning_experiments.ipynb`](../notebooks/tensorflow/05_deep_learning_experiments.ipynb).

---

### 3. Combining NST with Downstream Computer Vision Tasks (e.g., Semantic Segmentation)
* **Question**: *Can style transfer be constrained or combined with segmentation (e.g., background vs. foreground stylization)?*
* **Analysis**: Yes. Unconstrained NST often applies style indiscriminately across background and foreground subjects (e.g., distorting human faces or target objects). By integrating segmentation masks (e.g., U-Net, DeepLabV3, or rembg):
  - Spatial guidance masks $M_{content}$ can weight the content loss spatially.
  - Background can be stylized while foreground subject integrity is preserved, or vice versa.
* **Implementation in Repo**: Prototyped in [`notebooks/tensorflow/08_paper_based_segmentation_nst.ipynb`](../notebooks/tensorflow/08_paper_based_segmentation_nst.ipynb) utilizing `rembg`.

---

### 4. Cross-Domain Style Transfer Beyond 2D Visual Imagery
* **Question**: *Can NST concepts be extended beyond 2D digital images (e.g., audio, 3D, text)?*
* **Analysis**: 
  - **Audio Style Transfer**: Feasible via spectrogram representations (Short-Time Fourier Transform / STFT) where 1D/2D CNNs compute Gram matrices over timbre features while preserving pitch/rhythm contours.
  - **3D Neural Fields / Meshes**: Stylizing NeRFs (Neural Radiance Fields) or 3D Gaussian Splatting with 2D style priors.
  - **Text Style Transfer**: Typically handled with Latent Diffusion, Seq2Seq, or LLM prompt-engineering rather than Gram matrix optimization.
* **Decision**: While theoretically intriguing, maintaining focus on visual computer vision ensures high experimental rigor.

---

### 5. Research Focus & Scope
* **Question**: *Should our core focus remain strictly on visual NST benchmarking, or diverge into cross-modal domains?*
* **Recommendation**: Maintain a focused, rigorous scope on **Comparative Visual NST Across Modern Vision Backbones**:
  - Benchmark VGG-19 vs. EfficientNet vs. ConvNeXt vs. Vision Transformers (LeViT).
  - Evaluate style fidelity, texture preservation, semantic retention, and inference runtime.

---

### 6. Project Deliverable: Academic Benchmark vs. Application Prototype
* **Question**: *Is the primary goal an academic publication/benchmark or an end-user application?*
* **Recommendation**: A two-tiered milestone approach:
  1. **Phase 1 (Academic)**: Establish reproducible benchmark results, quantitative metrics (SSIM, LPIPS, Gram loss convergence), and qualitative comparisons.
  2. **Phase 2 (Product)**: Package the best-performing models into an interactive Streamlit or Gradio demo for real-time demonstration.

---

### 7. Alternative Vision Backbones Beyond VGG-19
* **Question**: *Which modern architectures should be benchmarked against VGG-19?*
* **Evaluated Architectures in this Repository**:
  - **VGG-19**: Classical baseline (Simonyan & Zisserman, 2014; Gatys et al., 2015).
  - **EfficientNet-B0 & EfficientNetV2-B0**: High parameter efficiency and scalable compound scaling (Tan & Le, 2019/2021).
  - **ConvNeXt**: Modernized pure convolutional network competing with ViTs (Liu et al., 2022).
  - **LeViT**: Fast hybrid Vision Transformer combining convolutions and attention (Graham et al., 2021).

---

### 8. Multi-Style Blending & Loss Weighting
* **Question**: *When blending multiple style images (e.g., 2 or 3 distinct styles), how should the loss and textures be combined?*
* **Formulation**:
  Multi-style transfer combines Gram matrices linearly with user-defined blending weights $w_k$ ($\sum_k w_k = 1$):
  $$\mathcal{L}_{style\_multi} = \sum_{l \in L} \sum_{k=1}^{K} w_k \cdot \frac{1}{4 N_l^2 M_l^2} \sum_{i,j} \left( G_{ij}^l(I_{gen}) - G_{ij}^l(I_{style\_k}) \right)^2$$
  Alternatively, style loss can be weighted adaptively across different spatial regions using spatial semantic masks.

---

## 📚 References
- Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). *A Neural Algorithm of Artistic Style*. arXiv:1508.06576.
- Johnson, J., Alahi, A., & Fei-Fei, L. (2016). *Perceptual Losses for Real-Time Style Transfer and Super-Resolution*. ECCV.
- Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML.
- Liu, Z., et al. (2022). *A ConvNet for the 2020s*. CVPR.
- Graham, B., et al. (2021). *LeViT: a Vision Transformer in disguise for high-speed inference*. ICCV.
