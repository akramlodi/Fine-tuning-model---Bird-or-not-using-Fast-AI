# Bird or Not Image Classifier (Fine-Tuned ResNet18)

This project builds a **binary image classifier** that distinguishes between **birds** and **forests** using **fastai** and **transfer learning**.

Instead of training a model from scratch, we **fine-tune a pretrained ResNet18 model** on a custom dataset collected automatically using the **Pexels API**.

---

## Features

* Automatic dataset creation using Pexels API
* Image preprocessing and resizing
* Transfer learning with pretrained ResNet18
* Fine-tuning on custom Bird vs Forest dataset
* Prediction on new images
* Runs easily on Google Colab

---

## Model Training Approach

This project uses **fine-tuning**:

1. Load pretrained **ResNet18**
2. Replace classification head
3. Train on Bird vs Forest dataset
4. Adjust pretrained weights using `fine_tune()`

Example training code:

```python
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)
```

This improves accuracy with a small dataset by leveraging features learned from ImageNet.

---

## Project Structure

```text
bird_or_not/
 ├── bird/
 └── forest/
```

Images are downloaded automatically before training.

---

## Requirements

Install dependencies:

```bash
pip install fastai fastcore fastdownload requests
```

---

## Setup

1. Get a free API key from:
   https://www.pexels.com/api/

2. Add your key inside the script:

```python
PEXELS_API_KEY = "YOUR_API_KEY"
```

---

## Training Pipeline

The notebook performs:

* dataset collection using Pexels API
* resizing images
* dataloader creation using fastai DataBlock
* fine-tuning ResNet18
* prediction on new images

Example prediction:

```python
is_bird,_,probs = learn.predict("bird.jpg")
```

Output:

```text
This is a: bird
Probability it's a bird: 0.98
```

---

## Run in Google Colab

Open the notebook in google colab.

---

## Future Improvements

* Add more image classes
* Export trained model (.pkl)
* Build web inference app
* Deploy using Streamlit or FastAPI

---
