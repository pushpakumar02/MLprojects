# Image Captioning using CNN & LSTM

This project implements an **Image Captioning Model** that combines **CNN (Xception)** for image feature extraction and **LSTM** for sequence generation. The model is trained on the **Flickr8k dataset** to generate meaningful captions for images.

## Table of Contents
- [Introduction](#introduction)
- [Basic Concepts](#basic-concepts)
- [Project Code Structure](#project-code-structure)
- [Setup and Installation](#setup-and-installation)
- [Training the Model](#training-the-model)
- [Generating Captions](#generating-captions)
- [Model Overview](#model-overview)
- [References & Acknowledgments](#references--acknowledgments)

## Introduction

This project utilizes **deep learning techniques** to automatically generate descriptive captions for images. The **CNN (Convolutional Neural Network)** extracts image features, while the **LSTM (Long Short-Term Memory)** network generates text descriptions based on those features.

## Basic Concepts

1. **Tokenization**: Converts text into smaller units (tokens) that a model can process. A token may be a word, subword, or even punctuation.
2. **Embedding**: Transforms tokens into numerical representations that a machine learning model can understand.
3. **CNN (Convolutional Neural Network)**: Used for image processing; extracts high-level features from images.
4. **LSTM (Long Short-Term Memory)**: A type of RNN (Recurrent Neural Network) specialized in handling sequential data, such as text.

## Project Code Structure

```
📂 Image-Captioning
│── 📁 Flickr8k_text/      # Contains image captions dataset
│── 📁 Flickr8k_Dataset/   # Contains image dataset
│── 📁 models/             # Stores trained models
│── 📁 output/             # Stores generated captions
│── main.py               # Data preprocessing & feature extraction
│── train.py              # Model training script
│── test.py               # Caption generation script
│── requirements.txt      # Python dependencies
│── README.md             # Project documentation
```

## Setup and Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-repo/image-captioning.git
   cd image-captioning
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Flickr8k Dataset:**
   - [Images & Captions Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k/)
   - Extract and place them inside `Flickr8k_text/` and `Flickr8k_Dataset/`

## Training the Model

Run the following command to start training:
```bash
python train.py
```
This will train the LSTM model on extracted CNN image features.

## Generating Captions

Use a trained model to generate captions for a new image:
```bash
python test.py --image <image_path>
```
