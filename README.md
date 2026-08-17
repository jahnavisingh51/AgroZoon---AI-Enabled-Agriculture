# AgroZoon

**AgroZoon** is a Flask web application for **AI-assisted agriculture**: crop recommendation, fertilizer guidance, and plant disease detection from leaf images. The UI uses a **dark green** theme and the app is now **login-protected** (you must sign in to access the main pages).

> **Disclaimer:** This is a proof-of-concept. Datasets and models are not certified for real farm decisions. Do not rely on outputs alone for agronomic or financial choices.

---

## Features

| Module | Description |
|--------|-------------|
| **Home** | Landing page with platform overview and links |
| **Crop recommendation** | Suggests crops from soil and weather-style inputs (ML when `crop_model.pkl` is present; heuristic fallback otherwise) |
| **Fertilizer** | Recommends nutrient adjustments from N/P/K and selected crop |
| **Disease detection** | Classifies leaf images with a PyTorch model when `plant_disease_model.pth` is available |
| **Authentication** | Login + Signup (bcrypt hashed passwords, session-based login) |
| **Profit optimizer** | Profit endpoint + UI template (if present in your build) |
| **Static guides** | Linked `AgriSens-web-app` explorer/guide assets |

---

## Requirements

- **Python 3.10, 3.11, or 3.12** (3.11 recommended)
- **pip** (recent)
- **Git** (optional, for clone)
- **~2 GB disk** for a typical venv including PyTorch CPU wheels

**Optional**

- **CUDA**-capable GPU: install a matching `torch` / `torchvision` build from [PyTorch](https://pytorch.org/get-started/locally/) instead of the CPU pins in `requirements.txt` if you need GPU inference.

---

## Quick start (new machine)

### 1. Open the project folder

If your folder is `d:\selling project\Agrozoon`, open PowerShell there.

### 2. Create and activate a virtual environment

**Windows (PowerShell)**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Model files (important)

The app **runs without** all models (crop and disease use fallbacks or graceful degradation), but for full behavior add:

| File | Typical location | Purpose |
|------|------------------|---------|
| `plant_disease_model.pth` | `models/plant_disease_model.pth` **relative to the process current working directory** | Disease CNN (see `app/app.py`) |
| `crop_model.pkl` | `app/models/crop_model.pkl` **relative to repo root** | Crop `RandomForest` classifier |

Train or obtain models using scripts in the repo root (e.g. `train_crop_model.py`, `train_disease_model.py`) when data and labels are available.

### 4. Run the application

Run from the **repository root** so paths and static files resolve consistently:

**Windows**

```powershell
.\run.ps1
```

or:

```powershell
.\venv\Scripts\python.exe app\app.py
```

**Linux / macOS**

```bash
./venv/bin/python app/app.py
```

Then open **http://127.0.0.1:5000/** (or the URL Flask prints).

### 5. Login flow

- Visit `/signup` to create an account.
- Then login at `/login`.
- After login, the **entire app** is accessible; without login, routes redirect to `/login`.

### 5. Production (Linux server)

Example with Gunicorn (installed via `requirements.txt`):

```bash
cd /path/to/AgroIntell
source venv/bin/activate
gunicorn -w 2 -b 0.0.0.0:8000 "app.app:app"
```

Adjust `PYTHONPATH` or working directory if your layout differs. On **Windows**, use `waitress` or run the dev server; Gunicorn is not supported natively on Windows.

---

## Dependency file layout

| File | Role |
|------|------|
| `requirements.txt` | **Canonical** list of packages for the whole project |
| `app/requirements.txt` | Includes the root file (`-r ../requirements.txt`) for convenience |

---

## Troubleshooting `pip install`

### PyTorch fails to install

Try the [official install command](https://pytorch.org/get-started/locally/) for your OS, then install the rest:

```bash
pip install Flask==3.0.3 Werkzeug==3.0.3 Jinja2==3.1.4 MarkupSafe==2.1.5 itsdangerous==2.2.0 click==8.1.7 blinker==1.8.2
pip install numpy==1.26.4 pandas==2.2.2 scipy==1.13.1 scikit-learn==1.5.0 joblib==1.4.2 Pillow==10.3.0 gunicorn==22.0.0
# Then torch / torchvision per PyTorch site
```

### `sklearn` import errors

Use the **`scikit-learn`** package only. Do not install the deprecated PyPI meta-package named `sklearn`.

---

## Project layout (high level)

```
AgroZoon/
├── app/
│   ├── app.py              # Flask entrypoint
│   ├── auth_routes.py       # Login / Signup / Logout routes
│   ├── models.py            # SQLAlchemy User model (SQLite by default)
│   ├── templates/          # Jinja2 HTML
│   ├── static/             # CSS, images, bundled web assets
│   ├── utils/              # Disease/fertilizer dictionaries, models, profit logic
│   └── models/             # Place crop_model.pkl here (see above)
├── models/                 # Optional: plant_disease_model.pth (cwd-relative)
├── AgriSens-web-app/       # Standalone static site (also under app/static/…)
├── requirements.txt
├── run.ps1                 # Windows helper to start the app
├── train_crop_model.py
├── train_disease_model.py
└── README.md
```

---

## Data & inspiration

- Crop / fertilizer / disease concepts and early structure trace back to **Harvestify** and related Kaggle/notebook workflows; see the original [Harvestify](https://github.com/Gladiator07/Harvestify) repository and its disclaimers.
- Dataset sources cited in the original project (Kaggle crop recommendation, plant disease dataset, etc.) apply when you train your own checkpoints.

---

## License

Check the `LICENSE` file in this repository if present; otherwise treat licensing as inherited from upstream projects you depend on.

---

## Contributing

Issues and pull requests are welcome. Keep changes focused; match existing style for templates and Python.
