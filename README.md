# MVGGT: Multimodal Visual Geometry Grounded Transformer for Multiview 3D Referring Expression Segmentation

<div align="center">

**Changli Wu**<sup>1,2,†</sup>, **Haodong Wang**<sup>1,†</sup>, **Jiayi Ji**<sup>1</sup>, **Yutian Yao**<sup>5</sup>,  
**Chunsai Du**<sup>4</sup>, **Jihua Kang**<sup>4</sup>, **Yanwei Fu**<sup>3,2</sup>, **Liujuan Cao**<sup>1,*</sup>

<sup>1</sup>Xiamen University, <sup>2</sup>Shanghai Innovation Institute, <sup>3</sup>Fudan University,  
<sup>4</sup>ByteDance, <sup>5</sup>Tianjin University of Science and Technology

<sup>†</sup>Equal Contribution, <sup>*</sup>Corresponding Author

[![Paper](https://img.shields.io/badge/Paper-Arxiv-b31b1b.svg)](https://arxiv.org/abs/2601.06874)
[![Project Page](https://img.shields.io/badge/Project-Website-blue.svg)](https://sosppxo.github.io/mvggt.github.io/)
[![Demo](https://img.shields.io/badge/Demo-HuggingFace-orange.svg)](https://huggingface.co/spaces/sosppxo/mvggt)
[![Weights](https://img.shields.io/badge/Weights-HuggingFace-yellow.svg)](https://huggingface.co/sosppxo/mvggt)

</div>

---

## 📢 News & Roadmap

This repository is the official implementation of **MVGGT**. We are currently preparing the code and data for release. Please stay tuned!

- [ ] **Release the MVRefer Benchmark** (Dataset & Evaluation scripts).
- [ ] **Release Training & Inference Code**.
- [ ] **Release Pre-trained Models**.
- [ ] **Release Interactive Demo Code** (Local version).

---

## 📖 Abstract

Most existing 3D referring expression segmentation (3DRES) methods rely on dense, high-quality point clouds, while real-world agents such as robots and mobile phones operate with only a few sparse RGB views and strict latency constraints. 

We introduce **Multi-view 3D Referring Expression Segmentation (MV-3DRES)**, where the model must recover scene structure and segment the referred object directly from sparse multi-view images. Traditional two-stage pipelines, which first reconstruct a point cloud and then perform segmentation, often yield low-quality geometry, produce coarse or degraded target regions, and run slowly. 

We propose the **Multimodal Visual Geometry Grounded Transformer (MVGGT)**, an efficient end-to-end framework that integrates language information into sparse-view geometric reasoning. Experiments show that MVGGT establishes the first strong baseline and achieves both high accuracy and fast inference, outperforming existing alternatives.

<div align="center">
  <img src="https://sosppxo.github.io/mvggt.github.io/resources/figure1.png" width="80%">
  <br>
  <em>Figure 1: Comparison of the proposed MV-3DRES task (bottom) against the traditional two-stage pipeline (top).</em>
</div>

## 🚀 Method: MVGGT

We propose the **Multimodal Visual Geometry Grounded Transformer (MVGGT)**, an end-to-end framework designed for efficiency and robustness.

![MVGGT Architecture](https://sosppxo.github.io/mvggt.github.io/resources/figure3.png)
*Figure 2: Architecture of MVGGT. It features a Frozen Reconstruction Branch (top) and a Trainable Multimodal Branch (bottom).*

> **Note:** For interactive 3D visualizations and video comparisons with other methods, please visit our [**Project Page**](https://sosppxo.github.io/mvggt.github.io/).

## 🚀 Demo Deployment

Follow these steps to deploy the interactive demo locally:

### 1. Clone the repository

```bash
git clone https://github.com/sosppxo/mvggt.git
cd mvggt
```

### 2. Create conda environment

Create and activate a new conda environment:

```bash
conda create -n mvggt python=3.12
conda activate mvggt
```

### 3. Install dependencies

Install the required packages for the demo:

```bash
pip install -r requirements_demo.txt
```

### 4. Download model weights and tokenizer

1. **Download pre-trained model weights**: Download from [Hugging Face](https://huggingface.co/sosppxo/mvggt) and update the `ckpt_path` in `demo_gradio.py` (line 608) to point to your checkpoint file.

2. **Download RoBERTa tokenizer**: The demo requires RoBERTa tokenizer. Download it using:

```bash
mkdir -p ckpts
python -c "from transformers import RobertaTokenizer; RobertaTokenizer.from_pretrained('roberta-base').save_pretrained('./ckpts/roberta-base')"
```

Or manually download from Hugging Face and place it in `./ckpts/roberta-base/`.

### 5. Launch the demo

Run the Gradio demo:

```bash
python demo_gradio.py
```

The demo will be available at `http://localhost:7860` (or the URL shown in the terminal). You can use the `share=True` option to create a public link.

### Usage

1. Upload multiple images or a video containing multi-view scenes
2. Enter a referring expression describing the target object
3. The model will generate 3D segmentation results that can be downloaded as GLB files

## 📝 Citation

If you find our work useful in your research, please consider citing:

```bibtex
@misc{wu2026mvggt,
  Author = {Changli Wu and Haodong Wang and Jiayi Ji and Yutian Yao and Chunsai Du and Jihua Kang and Yanwei Fu and Liujuan Cao},
  Title = {MVGGT: Multimodal Visual Geometry Grounded Transformer for Multiview 3D Referring Expression Segmentation},
  Year = {2026}
}
