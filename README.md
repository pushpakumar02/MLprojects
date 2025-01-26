# Image Captioning (CNN + LSTM)

This project uses **CNN (Xception)** and **LSTM** to generate captions for images from the **Flickr8k dataset**.

## Files

- `main.py`: Preprocesses data and extracts image features.
- `train.py`: Trains the image captioning model.
- `test.py`: Generates captions for new images using the trained model.

## Requirements

- Python 3.x
- TensorFlow, Keras, Pillow, Numpy

## Setup

1. Download **Flickr8k dataset** (images and captions).
2. Place them in `Flickr8k_text/` and `Flicker8k_Dataset/`.

## Training

Run `train.py` to train the model:

```bash
python train.py
```

## Testing

Generate captions for a new image:

```bash
python test.py --image <image_path>
```

## Model

- **Xception** extracts image features.
- **LSTM** generates captions based on those features.