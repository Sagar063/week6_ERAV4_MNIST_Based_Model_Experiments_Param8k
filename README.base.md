MNIST Multi-Model Experiments (PyTorch)

Baselines → BN/Dropout/GAP → Aug+StepLR to reach ≥99.4% in ≤15 epochs with ≤8k params.

Repository Structure

train.py — runner (CLI families, model summary, tqdm, plots, metrics, checkpoints)

model1.py — baselines: Big (~194k), Light (~10.7k)

model2.py — Light + BN / DO / BN+DO / BN+DO+GAP

model3.py — capacity, pooling fix, tiny+Aug+StepLR target

results/ — per-run CSVs, plots, confusion matrices, summary.csv

checkpoints/ — best weights per experiment

Setup
pip install -r requirements.txt


If you have a CUDA GPU, use the matching Torch/TorchVision wheels (see requirements.txt notes).

Usage
# run all
python train.py --epochs 15 --batch_size 128

# run only model families
python train.py --model1 --epochs 15
python train.py model2 --epochs 15
python train.py --model3 --epochs 15

# run specific experiments
python train.py --model2 --only m2_light_bn_do_gap --epochs 15


Outputs

Metrics per run: results/metrics_<exp>.csv

Plots: results/plots/acc_<exp>.png, results/plots/loss_<exp>.png, results/plots/cm_<exp>.png

Best checkpoints: checkpoints/<exp>/best.pt

Consolidated: results/summary.csv

Dataset & Normalization

MNIST (60k train / 10k test, 28×28 grayscale).
Normalization is computed on the training set only and applied to both train and test.

<!-- DATA_STATS -->
Experiments (Objectives → Variants → Results → Analysis)
Model1 — Baselines

Objective: Establish reference points for capacity vs accuracy with one large and one compact CNN.

Variants

m1_big — deeper CNN (~194k params).

m1_light — compact CNN (~10.7k params).

Auto-generated results and analysis

<!-- MODEL1_RESULTS -->
Model2 — Light Model Improvements

Objective: Improve the light baseline’s stability and generalization; reduce parameters via GAP.

Definitions

Batch Normalization (BN): normalizes mini-batch activations to stabilize/accelerate training; applied after conv layers (not the final classifier).

Dropout (DO): randomly zeroes activations during training to reduce overfitting.

Global Average Pooling (GAP): replaces fully connected layers with per-channel spatial averaging → large parameter reduction and regularization.

Variants

m2_light_bn, m2_light_do, m2_light_bn_do, m2_light_bn_do_gap

Auto-generated results and analysis

<!-- MODEL2_RESULTS -->
Model3 — Advanced Tweaks Toward Target

Objective: Hit ≥99.4% consistently (last few epochs), ≤15 epochs, ≤8k parameters.

Techniques

Capacity tuning (judiciously increasing channels)

Correct pooling placement (after sufficient convs)

Data augmentation (rotation/affine/perspective/erasing)

StepLR scheduling (refine late-epoch learning)

Variants

m3_capacity, m3_poolfix, m3_aug_steplr (tiny + Aug + StepLR, target)

Auto-generated results, target checks, and analysis

<!-- MODEL3_RESULTS -->
Consolidated Results
<!-- CONSOLIDATED_RESULTS -->
Observations & Learnings

Capacity vs. accuracy trade-off (e.g., m1_big vs m1_light)

BN/DO effects (stability/overfitting vs m1_light)

GAP efficiency (parameter cuts with minimal accuracy loss)

Augmentation + StepLR impact (pushing tiny model to the target)

<!-- OBSERVATIONS -->
Reproducibility

Seed used: 42

Same normalization stats applied to both train & test

Environment:

<!-- ENV_INFO -->
References

PyTorch: https://pytorch.org/docs/stable/

torchvision MNIST: https://pytorch.org/vision/stable/datasets.html#mnist