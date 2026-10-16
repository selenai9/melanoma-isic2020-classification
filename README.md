# Cost-Sensitive Melanoma Detection: Handcrafted-Feature Machine Learning vs Convolutional Networks, with and without Clinical Metadata

Introduction to Machine Learning Summative

Author: **Selena ISIMBI**

| Resource | Link |
|---|---|
| Code (this repository) | https://github.com/selenai9/melanoma-isic2020-classification |
| Data, saved models and logs (Google Drive, view access) | https://drive.google.com/drive/folders/1-hHbnQsSVu1c-4T3rmfo5iOxOHYJtI4L?usp=sharing |
| Video presentation | [Video link] |

## Research question

Which model achieves the highest recall (sensitivity) at 90% specificity under severe class imbalance, and do clinical metadata (age, sex, lesion site) improve it?

## Dataset

SIIM-ISIC 2020 (ISIC 2020 challenge): 33,126 dermoscopic images from 2,056 patients, of which 584 (1.76%) are malignant melanomas.

- Images: the resized public Kaggle mirror `nischaydnk/isic-2020-jpg-256x256-resized` (downloaded automatically by the notebook with `kagglehub`).
- Clinical metadata: the official `train.csv` from the ISIC 2020 challenge (age, sex, lesion site, diagnosis).
- **Raw data are not stored in this repository.** The zipped raw dataset is in the Google Drive folder above.
- The split is **by patient** (23,885 / 4,709 / 4,532 images for train / validation / test), so no patient appears in more than one set.

## Methods

- **Traditional machine learning (scikit-learn):** logistic regression, random forest and histogram gradient boosting on 48 handcrafted ABCD-style features (colour, shape, asymmetry, border, texture) with and without 12 clinical metadata features; logistic regression on frozen EfficientNetB0 embeddings.
- **Deep learning (TensorFlow/Keras):** a CNN trained from scratch, a classifier head on frozen EfficientNetB0 embeddings, a two-phase fine-tuned EfficientNetB0 and an image-plus-metadata fusion model (Functional API), all with `tf.data` input pipelines.
- **Experiments (9, each varied systematically and logged):** E1 logistic regression; E2 random forest and gradient boosting; E3 imbalance handling; E4 handcrafted features plus metadata; E5 baseline CNN; E6a CNN with cosine learning-rate schedule; E7a/E7b frozen EfficientNetB0 embeddings; E8 fine-tuned EfficientNetB0; E9 fusion with metadata.
- **Evaluation:** ROC-AUC, PR-AUC and recall at 90% specificity, with patient-level bootstrap 95% confidence intervals and paired bootstrap comparisons. Decision thresholds are fixed on the validation set and applied unchanged to the test set, which is used once. The notebook also contains learning curves, confusion matrices, ROC and precision-recall curves, a bias-variance discussion, calibration analysis and error analysis.

## Results (test set, thresholds fixed on validation)

| Model | ROC-AUC | PR-AUC | Recall at 90% specificity |
|---|---|---|---|
| Gradient boosting + metadata (best traditional model, E4) | 0.848 | 0.109 | 0.438 |
| Fine-tuned EfficientNetB0 (best deep learning model, E8) | 0.886 | 0.219 | 0.648 |
| EfficientNetB0 + metadata fusion (E9) | 0.886 | 0.203 | 0.629 |

Fine-tuning clearly improved on the from-scratch CNN, and clinical metadata gave no measurable gain once the image network was fine-tuned (paired bootstrap). The full table of all experiments is in `results_table_clean.csv`.

## Repository contents

| File | Description |
|---|---|
| `summative_assignment.ipynb` | The full notebook: data, experiments, evaluation, error analysis |
| `requirements.txt` | Package versions used (Python 3.13 on Google Colab) |
| `results_table_clean.csv` | Results of all experiments (validation) and the test evaluation |
| `README.md` | This file |

## How to reproduce

1. Open the notebook in Google Colab and select a **T4 GPU** runtime.
2. Create a Drive folder `MyDrive/melanoma_project/` and place the official `train.csv.zip` in it. To skip retraining, also copy the `checkpoints/`, `logs/` and `models/` folders from the shared Drive folder above.
3. Run all cells from top to bottom. All random seeds are fixed (`SEED = 42`).
4. The slow training steps are switched off by default and reload the saved weights from Drive. Set `RUN_E5_TRAINING`, `RUN_E6A_TRAINING`, `RUN_E8_TRAINING` or `RUN_E9_TRAINING` to `True` to retrain that model (about 15 to 40 minutes each on a T4). Retrained numbers may differ slightly in the third decimal because GPU training is not fully repeatable.
5. To install the same package versions elsewhere: `pip install -r requirements.txt`.

Experiment metrics are logged to `results_table.csv` and to offline Weights & Biases runs (`logs/wandb`); TensorBoard logs are in `logs/tb`. These log folders are in the Google Drive folder.

## Saved models

In the Google Drive folder, under `models/`:
- `best_traditional_hgb_meta.joblib` with `best_traditional_feature_columns.json`: gradient boosting model on handcrafted and metadata features.
- `e8_effnetb0_finetuned.keras`: fine-tuned EfficientNetB0 (best deep learning model).
- `e9_fusion.keras`: image-plus-metadata fusion model.

## Limitations

- The test set holds only 105 malignant images, so confidence intervals are wide; subgroup results are indicative only.
- ISIC 2020 has no skin-tone labels, so performance across skin types could not be evaluated.
- Images are the 256x256 resized versions, so fine texture detail is reduced.
- The handcrafted features rely on simple Otsu segmentation, which can fail on pale or low-contrast lesions.
- This is an experimental comparison, not a clinical validation study.
