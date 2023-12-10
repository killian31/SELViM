# SSL-Vision

This project aims to train a Vision Transformer with a diffusion head to perform
inpainting on images using self-supervised learning.

## Installation & Usage

### Installation

Clone the repo:

```bash
git clone https://github.com/killian31/SSL-Vision.git
```

Install dependencies:

```bash
cd SSL-Vision
pip install -r requirements.txt
```

### Example Usage

```bash
python3 train.py --n_patches 7 --n_blocks 4 --hidden_d 16 --n_heads 4 --out_d 10 --n_epochs 20 --lr 0.001 --batch_size 128 --save_model_freq 1 --save_dir test_model --eval_freq 1
```
