# PushFit: PushFit: Push-Up Form Classification via Pose Estimation

**CS 231N: Deep Learning for Computer Vision**  
**Created by**: Ramya Iyer

## Overview

PushFit is a real-time, vision-based system created to analyze human push-up form using pose estimation and classification models. It supports both classical (SVM, Random Forest) and temporal (1D CNN, LSTM, GRU) classification techniques. The system extracts pose keypoints from short push-up videos using MediaPipe Pose and MoveNet, then classifies the form as either correct or incorrect.

## Implemented Pipeline

1. **Input**: Short `.mp4` video of a person performing at least one push-up.
2. **Pose Estimation**:
   - **MediaPipe Pose**: Outputs 33 keypoints per frame, includes visibility scores.
   - **MoveNet**: Outputs 17 keypoints per frame, includes confidence scores.
3. **Classification**:
   - **Classical Models**: Support Vector Machines (SVM), Random Forest (RF)
   - **Temporal Models**: 1D CNN, LSTM, GRU
4. **Output**: Binary label - "correct" or "incorrect" push-up form.
5. **Extract Features**: Incorrect push-up forms extract features from pose estimation methods.
6. **Feedback**: Features passed through `Gemini 1.5 Flash` for more precise feedback.

## Dataset

- **Total videos**: 265
  - 140 Correct Form
  - 125 Incorrect Form
- **Sources**:
  - [LSTM Push-Up Classification Dataset](https://www.kaggle.com/datasets/mohamadashrafsalama/pushup)
  - [Workout / Exercises Videos](https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video)
  - [Gym Workout Exercises](https://www.kaggle.com/datasets/philosopher0808/gym-workoutexercises-video)

All videos are in `.mp4` format. Pose keypoints are extracted and stored in `.json` format for model input.

## Pose Estimation Details

- **MediaPipe Pose**: 33 keypoints per frame, optimized for real-time performance.
- **MoveNet**: 17 keypoints per frame, higher accuracy but slower due to preprocessing.

## Classification Models

### Classical Methods

- **SVM**: Uses RBF kernel with probability estimates.
- **Random Forest**: 100 trees, handles nonlinear boundaries well and resists overfitting.

### Temporal Models

- **1D CNN**: Extracts temporal features, trained for 100 epochs with early stopping.
- **LSTM**: Sequence length of 50 frames, uses Binary Cross Entropy loss.
- **GRU**: Similar to LSTM but with fewer parameters; trained for 100 epochs.

## Experiments

### Pose Estimation Runtime

- MediaPipe Pose: ~667 seconds
- MoveNet: ~1236 seconds

### Validation Accuracy (No Cross-Validation)

| Model         | MediaPipe Pose | MoveNet    |
| ------------- | -------------- | ---------- |
| SVM           | 71.70%         | 56.60%     |
| Random Forest | 81.13%         | **90.57%** |
| 1D CNN        | **88.68%**     | 69.81%     |
| LSTM          | 75.47%         | 52.83%     |
| GRU           | 79.25%         | 62.26%     |

### Validation Accuracy (5-Fold Cross-Validation)

| Model         | MediaPipe Pose | MoveNet    |
| ------------- | -------------- | ---------- |
| SVM           | 74.34%         | 58.49%     |
| Random Forest | **82.64%**     | **87.55%** |
| 1D CNN        | 79.25%         | 61.89%     |
| LSTM          | 81.51%         | 58.11%     |
| GRU           | 75.85%         | 58.87%     |

## Conclusion

- **Best Combination**: MoveNet + Random Forest (90.57% validation accuracy)
- MediaPipe + 1D CNN also performs competitively at 88.68%
- Random Forest consistently performs well across different estimators
- Cross-validation confirms model generalization

# CS 231N Poster

![CS 231N Final Poster](visuals/CS%20231N%20Poster.jpg)

### Additional Notes

Tensorflow model not rendering, run this:

```
rm -rf /var/folders/ct/h6jfsjs122zcn6rz6mz_7pfc0000gn/T/tfhub_modules/
```
