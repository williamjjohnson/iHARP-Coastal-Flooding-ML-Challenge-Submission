# Predicting Coastal Flooding Events - Scientific Out-of-Distribution Challenge

## GAF-CNN flood prediction

This submission uses a finetuned ResNet18 model to predict flooding days using 7-day hourly data represented as Gramian Angular Summation fields. The model outputs 14 binary predictions (one-hot encoding) for the forecast period following each 7-day input window. Flooding days are defined using the threshold calculation approach used in the example model (mean + 1.5\*std). Date windows and stations used for testing are excluded from the training dataset.

## Model Architecture

- **Base Model**: ResNet18 (pretrained on ImageNet)
- **Input**: 168×168 grayscale GAF images (converted to RGB)
- **Output**: 14 binary predictions (multi-label classification)
- **Loss Function**: BCEWithLogitsLoss
- **Training**: Fine-tuning with early stopping (patience=3)

## Important processing steps for this model include

1. Linear interpolation for missing hourly values

```
imputer = InterpolationImputer()
X_train_imputed = imputer.fit_transform(X_train.squeeze())
```
1. Creating Gramian Angular Summation Field for 7 day training windows

```
gaf = GramianAngularField()
X_train_GAF = gaf.fit_transform(X_train_imputed)
```
1. Representing gramian angular field as a 3 channel image scaled from 0 to 255 for compatibility with resnet pretraining

```
def get_x(i):
    gaf_scaled = ((X_train_GAF[i] + 1) * 127.5).astype(np.uint8)
    img = Image.fromarray(gaf_scaled, mode='L')
    return PILImage.create(img).convert('RGB')
```

The model achieved high accuracy while testing on out of distribution results (for data from Lewes, Fernandina Beach, and The Battery):

```
=== Confusion Matrix ===
TP: 404 | FP: 23 | TN: 42 | FN: 119

=== Metrics ===
Accuracy: 0.759
F1 Score: 0.851
MCC: 0.294
```

## Submission Contents

- `model.py`: Complete training and inference pipeline
- `model.pkl`: Trained FastAI learner (serialized with dill)
- `requirements.txt`: Python package dependencies
- `README.md`: This file
