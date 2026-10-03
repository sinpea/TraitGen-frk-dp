## Installation

Create and activate the Conda environment:

```bash
conda create -n gen python=3.8 -y
conda activate gen
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

## Training

To start training, run:

Example:

```bash
torchrun --nproc_per_node=1 main.py --data_root "<path to dataset>" --ann_dir "<path to COCO annotations>" --batch_size 4 --epochs 15 --lr 1e-4 --decoder_model "Qwen/Qwen3-1.7B-Base"
```

By default, the training logs and model checkpoints are saved in the `output/` directory.
