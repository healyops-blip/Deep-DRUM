# Deep-DRUM

Deep-DRUM is a hybrid physical-computational framework for rapid intraoperative histopathology of thick tissues. It converts label-free DRUM images into high-fidelity virtual H&E images. Because H&E sectioning destroys the original tissue volume and prevents pixel-wise ground-truth pairing, Deep-DRUM determines a 20 μm effective DRUM imaging depth in brain tissue and establishes physical DRUM/H&E pairing from the same thick specimen over the same axial range. Hierarchical registration then provides cellular-level alignment, followed by deep-learning virtual H&E staining.

## Installation

Install the shared dependencies for data construction and translation.

```bash
cd deep-drum
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 1. Data Construction

Register DRUM/H&E images and generate paired A|B data.

```bash
cd data_construction
python main.py
cd ..
```

## 2. Translation Training

Train supervised Pix2Pix with paired images.

```bash
cd translation
DATASET_NAME="<dataset_name>"
EXPERIMENT_NAME="<experiment_name>"
python train.py \
  --dataroot "../data_construction/output/${DATASET_NAME}/fold_AB" \
  --name "$EXPERIMENT_NAME" \
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --input_nc 3 --output_nc 3 --netG unet_256 --norm batch \
  --load_size 286 --crop_size 256 --batch_size 8 \
  --n_epochs 100 --n_epochs_decay 100 \
  --lr 0.0002 --beta1 0.5 --lambda_L1 100 --gan_mode vanilla \
  --display_id -1 --no_html --gpu_ids 0
cd ..
```

## 3. Paired-Image Translation Testing

Test paired images with trained weights.

```bash
cd translation
DATASET_NAME="<dataset_name>"
EXPERIMENT_NAME="<experiment_name>"
python test.py \
  --dataroot "../data_construction/output/${DATASET_NAME}/fold_AB" \
  --name "$EXPERIMENT_NAME" \
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --netG unet_256 --norm batch \
  --phase train --epoch latest \
  --load_size 256 --crop_size 256 --num_test "<number_of_test_images>" \
  --eval --gpu_ids 0 \
  --results_dir ../results
cd ..
```

## 4. Accelerated Large-Image Translation

Run batched inference on image patches and stitch the results.

- Input: `translation/datasets/stitch/test.jpg`
- Weights: `translation/checkpoints/<experiment_name>/latest_net_G.pth`
- Batch size: 192
- Mixed precision: enabled
- Output: `translation/datasets/stitch/output/test_Hp.png`

Run the pipeline:

```bash
cd translation
python test_stitch_batch.py
cd ..
```

## Sources, Licensing, and Citation

This section records code provenance, licensing, and citation requirements.

### Code Provenance and Licensing

The table identifies the license scope of upstream code.

| Code | Source | License |
| --- | --- | --- |
| Translation | [PyTorch CycleGAN and Pix2Pix by Jun-Yan Zhu and Taesung Park](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) | BSD; see [translation/LICENSE](translation/LICENSE) |

### Citation Format

The Deep-DRUM manuscript has not yet been published and currently has no journal, publication year, or DOI. Cite it as an unpublished manuscript until the final publication record is available:

```bibtex
@unpublished{li_deep_drum,
  author={Li, Hui and Wen, Zonghua and Yang, Xu and Cheng, Yulong and Li, Jian and Zhang, Ruikai and He, Yuezhi and Zheng, Wei and Ye, Shiwei},
  title={Deep-DRUM: Virtual H\&E Staining of Label-Free Thick Tissues for Rapid Intraoperative Histopathology},
  note={Unpublished manuscript.}
}
```

When citing this code, also state the repository URL and the exact version used in the main text or software acknowledgments. Replace this temporary citation with the formal publication record after the manuscript is published or the software receives a DOI.
