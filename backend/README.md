# Agriculture Analysis — Backend

Flask API that serves a trained CNN and returns a chilli disease diagnosis
to the `index.html` frontend.

## Folder structure

```
backend/
  app.py              # Flask app with the /predict endpoint
  requirements.txt
  model/
    chilli_model.h5   # <-- put your trained Keras model here
```

## Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Plug in your trained model

1. Train (or export) your CNN as a Keras `.h5` file, e.g.:
   ```python
   model.save("model/chilli_model.h5")
   ```
2. Make sure `CLASS_NAMES` in `app.py` is in the **same order** as your
   training generator's class indices:
   ```python
   print(train_generator.class_indices)
   # e.g. {'anthracnose': 0, 'bacterialSpot': 1, 'cercospora': 2, ...}
   ```
   Reorder the `CLASS_NAMES` list in `app.py` to match exactly.
3. If you trained on a different input size than 224x224 (e.g. VGG16
   default), update `IMG_SIZE` in `app.py` to match.
4. If your model is PyTorch instead of Keras, swap out `load_model()`
   and `predict()` in `app.py` — everything else (routes, response
   shape) stays the same.

## Run it

```bash
python app.py
```

Server starts at `http://localhost:5000`. Check it's alive:

```bash
curl http://localhost:5000/health
```

## Connect the frontend

Open `index.html` — it's already set up to POST to
`http://localhost:5000/predict` and falls back to the built-in
in-browser heuristic if the backend isn't reachable, so you can
develop the UI and the model independently.

## API

**POST** `/predict`
Multipart form-data, field name `image`.

Response:
```json
{
  "label": "anthracnose",
  "confidence": 91.4,
  "severity": 75,
  "raw_scores": { "healthy": 2.1, "anthracnose": 91.4, "...": 0.0 }
}
```

`label` must be one of the keys already defined in the `CONDITIONS`
object inside `index.html`'s `<script>` tag: `healthy`, `leafCurl`,
`anthracnose`, `bacterialSpot`, `powderyMildew`, `cercospora`. Add
more classes on both sides if you train against a larger label set.

## Deploying

For a real deployment (not just local dev), put this behind something
like Gunicorn + Nginx, or deploy to a platform such as Render, Railway,
or a small VM — `debug=True` in `app.py` should be turned off first.
