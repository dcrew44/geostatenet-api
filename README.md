# GeoStateNet API 🌐

FastAPI micro‑service that wraps a trained **GeoStateNet** model so other tools (the Chrome extension or any script) can predict US states from Street‑View‑like images over HTTP.

It is primarily intended to power the ***[GeoStateNet Chrome extension](https://github.com/dcrew44/geostatenet-extension)***, which overlays predictions onto live GeoGuessr rounds, but you can call the endpoints from any HTTP client or script.

[GeoStateNet base repo »](https://github.com/dcrew44/GeoStateNet)

---

## 🚀 Quick start

### 1  Clone & enter the repo

```bash
git clone https://github.com/dcrew44/geostatenet-api.git
cd geostatenet-api
```

### 2  Create and activate a Python virtual‑env

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\activate
```

### 3  Install requirements

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **CUDA users** – replace the `torch` / `torchvision` lines in `requirements.txt` with the matching CUDA wheels (e.g. `+cu118`) if you want GPU inference.

### 4  Add model weights

After downloading the checkpoint file from the release page:

```bash
mkdir -p weights  # if it doesn't exist
cp ~/Downloads/best_model.pth weights/
```

### 5  Run the server

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

You should see:

```text
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Open your browser:

* **`/`** → basic status ("GeoStateNet API is up 🎉")
* **`/docs`** → interactive Swagger UI
* **`/redoc`** → alternative docs

---

## 📑 Endpoints

| Route               | Method                       | Description                                         |
| ------------------- | ---------------------------- | --------------------------------------------------- |
| `/`                 | GET                          | Health check                                        |
| `/model/status`     | GET                          | Model metadata (architecture, checkpoint timestamp) |
| `/predict`          | POST (`multipart/form-data`) | Single 224×224 image (`file=...`)                   |
| `/predict_base64`   | POST (`application/json`)    | `{ "image": "<base64>" }`                           |
| `/predict_panorama` | POST (`multipart/form-data`) | 4 files: `north`, `east`, `south`, `west`           |

All responses share this JSON schema:

```json
{
  "predictions": [
    {"rank": 1, "state": "California", "state_abbrev": "CA", "probability": 82.95},
    {"rank": 2, "state": "Nevada", "state_abbrev": "NV", "probability": 5.11},
    ...
  ],
  "top_prediction": "California",
  "top_prediction_abbrev": "CA",
  "confidence": 82.95
}
```

## 🛠 Troubleshooting

| Symptom                                             | Fix                                                                                                   |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: uvicorn`                      | Activate your virtual‑env or run `pip install uvicorn[standard]`.                                     |
| `RuntimeError: shape mismatch` when loading weights | Ensure the checkpoint matches the model architecture and is named `best_model.pth` inside `weights/`. |
| CORS errors from browser extension                  | The API enables `*` CORS by default; double‑check the extension is hitting the same host & port.      |

---

## 📜 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.


