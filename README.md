# Thumbs Up / Thumbs Down Object Detector

A small computer-vision project that collects webcam images of "Thumbs Up" and "Thumbs Down" hand gestures, labels them with bounding boxes, and trains a YOLOv8 object detector (via `keras_cv`) to recognize the two gestures in images.

## Overview

The project is split into two main stages, driven by a Jupyter notebook and a training script:

1. **Data collection & labeling** — captures images from a webcam using OpenCV, resizes them, and annotates them with [labelImg](https://github.com/tzutalin/labelImg) to produce Pascal VOC-style XML bounding-box annotations.
2. **Model training** — loads the labeled images/annotations into a `tf.data` pipeline, augments them, and fine-tunes a `YOLOV8Detector` (from `keras_cv`, using a `yolo_v8_xs_backbone_coco` backbone) to detect `ThumbsUp` and `ThumbsDown` gestures.

## Project Structure

Running the notebook creates the following workspace layout:

```
Workspace/
└── GoodOrBad/
    ├── Images/
    │   ├── CollectedImages/     # raw webcam captures (uuid-named .jpg files)
    │   └── ProcessedImages/     # resized (640x640) copies used for training
    └── Annotations/             # Pascal VOC XML files produced by labelImg
labelImg/                        # cloned labelImg GUI tool (for annotating images)
model.h5                         # best checkpoint saved during training (by mAP)
```

## Requirements

- Python 3.11 (project was built/tested inside a `venv`)
- A connected webcam (for image collection)
- Key packages:
  - `opencv-python`
  - `uuid`
  - `tensorflow` / `tensorflow-macos` (Apple Silicon) with GPU support
  - `keras`, `keras_cv`, `keras-core`
  - `numpy`, `pillow`, `matplotlib`, `lxml`, `pycocotools`
  - `pyqt5` and `lxml` (build dependencies for `labelImg`)

> **Note:** There is a known dependency conflict — `tensorflow-macos` pins `tensorflow==2.16.2`, but a later step installs `tensorflow 2.20.0`, which pip flags as incompatible. Consider pinning your `tensorflow`/`keras`/`keras_cv` versions together in a fresh virtual environment to avoid version drift.

## Usage

### 1. Set up the environment

```bash
python -m venv venv
source venv/bin/activate
pip install opencv-python uuid
pip install pillow numpy keras matplotlib lxml keras_cv pycocotools
pip install pyqt5 lxml
```

### 2. Collect images

Run the notebook cells that:
- Create the `Workspace/GoodOrBad/...` folder structure.
- Open the webcam (`cv2.VideoCapture(0)`) to confirm it works.
- Loop over the `labels` list (`ThumbsUp`, `ThumbsDown`) and capture `numimg` (default 100) images per label, saving each as a UUID-named `.jpg` in `CollectedImages/`.

A single-shot capture cell is also included for grabbing one-off images.

### 3. Resize images

The notebook resizes every collected image to `640x640` and writes the result to `ProcessedImages/`, which is the resolution the YOLO model expects.

### 4. Label the images

`labelImg` is cloned from GitHub and built/launched:

```bash
git clone https://github.com/tzutalin/labelImg
cd labelImg && make qt5py3
python labelImg/labelImg.py
```

Use the labelImg GUI to draw bounding boxes around the hand gesture in each processed image and save the annotation as a Pascal VOC XML file into `Workspace/GoodOrBad/Annotations/` (filename matching the image name).

### 5. Train the detector

The training script/notebook:

1. Defines `class_ids = ["ThumbsDown", "ThumbsUp"]` and a `class_mapping`.
2. Parses each XML annotation (`read_annotation_file`) into bounding boxes and class IDs.
3. Builds parallel arrays of image paths, boxes, and classes (`get_x_and_y`) and wraps them in a `tf.data.Dataset`, holding out 10% for validation.
4. Converts the dataset into the `{"images": ..., "bounding_boxes": {"classes": ..., "boxes": ...}}` format expected by `keras_cv`, and applies augmentation (`RandomFlip`, `RandomShear`, `JitteredResize`).
5. Builds a `YOLOV8Detector` with a `yolo_v8_xs_backbone_coco` backbone, compiled with the Adam optimizer, `binary_crossentropy` classification loss, and `ciou` box loss.
6. Trains for 20 epochs using a custom `EvaluateCOCOMetricsCallback` that computes COCO-style mAP/recall each epoch and saves the model to `model.h5` whenever mAP improves.
7. Visualizes predictions on the validation set with `keras_cv.visualization.plot_bounding_box_gallery`.

```python
yolo.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[EvaluateCOCOMetricsCallback(val_ds, "model.h5")],
)
```

### 6. Run inference

Load the saved model and predict on new images/validation data:

```python
loaded_model = keras.models.load_model("model.h5")
predictions = loaded_model.predict(images[0:1])
```

> The current notebook has a cell that calls `loaded_model.predict(...)` without first loading the model — make sure to load `model.h5` (as shown above) before running inference.

## Results

Over 20 training epochs, validation mAP fluctuated in the 0.15–0.52 range (mAP@IoU=50 reaching 1.00 in several epochs), with the best checkpoint saved automatically to `model.h5` whenever validation mAP improved. Results are noisy due to the small dataset size (200 images total, 20 held out for validation) and would benefit from more data and longer/more stable training.

## Known Issues

- Dependency conflict between `tensorflow-macos` and a separately installed `tensorflow` version.
- Model is saved in the legacy HDF5 (`.h5`) format; Keras recommends switching to the native `.keras` format (`model.save("model.keras")`).
- A few `labelImg` sessions include "Cancel creation" and duplicate image/annotation log lines, suggesting some frames were skipped or re-saved during labeling — double-check the `Annotations/` folder for completeness before training.
- Small dataset (100 images per class) — consider collecting more varied images (lighting, background, hand position) to improve generalization.

## License

Add your preferred license here.
