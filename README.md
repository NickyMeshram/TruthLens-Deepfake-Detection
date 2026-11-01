# TruthLens — Deepfake Image Screening Demo

A Streamlit learning project that uses a pretrained image-classification model to estimate whether a face image resembles a real or manipulated image.

## Important limitations

- This first version uses an existing pretrained model; it does **not** train a model from scratch.
- It is an image-level screening demo, not a reliable forensic tool and not a video detector.
- Model scores are not proof that an image is real or fake. Performance may vary on unfamiliar generation/manipulation methods, compression, and image quality.
- Avoid uploading private or sensitive images.

## Run locally

Use Python 3.11 or 3.12 for the smoothest compatibility with the machine-learning dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the model weights and may take a few minutes. An internet connection is required for the initial model download.

## Tech stack

- Python
- Streamlit
- Hugging Face Transformers
- PyTorch
- Pretrained deepfake image-classification model: `KoreaPeter/ms-eff-gcvit-deepfake-b0-kodf`

## Suggested next steps

1. Test on a small set of clearly labelled real and manipulated face images that you have permission to use.
2. Record results and inspect false positives and false negatives.
3. Add a small evaluation report with precision, recall, F1 score, and a confusion matrix.
4. Add frame sampling and aggregation only if extending to video, and evaluate that separately.
