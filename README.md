# Face Mask Detection System

## About the Project
This project focuses on building a simple and practical system that can tell whether a person in an image is wearing a face mask or not.

Instead of checking manually, the idea is to automate the process using a trained deep learning model. The system takes an image as input and returns a clear result along with a confidence score and a suggested action.

---

## What it does
- Classifies images into:
  - Mask
  - No Mask
- Returns a confidence score
- Suggests a simple decision (allow / deny)
- Can be accessed through an API

---

## How it works
The project is split into a few main parts:

- Preparing and cleaning the dataset  
- Training a model (MobileNetV2)  
- Evaluating performance  
- Running inference on new images  
- Serving everything through a FastAPI backend  

---

## Tools used
- Python  
- PyTorch  
- OpenCV  
- FastAPI  
- Docker (for deployment)  

---

## Dataset
We used a public face mask dataset from Kaggle that contains around 12K images.

The data is split into:
- Training
- Validation
- Testing

Some basic augmentation is applied to improve generalization.

---

## Example output
```bash
{
  "status": "mask_on",
  "confidence": 0.96,
  "action": "Allow entry"
}
```

---

## Notes
- Results depend on image quality, lighting, and angles
- The model is trained on a limited dataset, so it may not cover all real-world cases
- This project is mainly for learning and demonstration purposes

---

## Possible improvements
- Add real-time webcam detection
- Detect incorrect mask usage
- Improve dataset diversity
- Optimize performance for faster inference

---

This project was developed as part of a university course, with work divided across data preparation, modeling, evaluation, API development, and deployment.