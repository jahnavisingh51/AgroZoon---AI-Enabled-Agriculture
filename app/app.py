from datetime import timedelta

from flask import Flask, render_template, request, redirect, jsonify, url_for, session
from markupsafe import Markup
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
from utils.profit_optimizer import (
    heuristic_top3_crops,
    rank_profit_table,
    why_this_crop_is_profitable,
)
from utils.profit_optimizer_predictor import (
    CROP_DISPLAY,
    calc_profit,
    predict_yield_and_future_price_ml,
    suggest_crop_for_better_profit,
)
import io
import os
from PIL import Image
import joblib  # for loading/saving ML models

from auth_routes import auth_bp
from models import User, db


# ------------------------- LOADING THE TRAINED MODELS -----------------------------------------------

# Disease model - lazy loaded (requires PyTorch + plant_disease_model.pth)
disease_model = None
disease_classes = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]

_app_dir = os.path.abspath(os.path.dirname(__file__))
_repo_dir = os.path.abspath(os.path.join(_app_dir, os.pardir))
disease_model_candidates = [
    # allow override from environment for custom deployments
    os.environ.get("DISEASE_MODEL_PATH", "").strip(),
    # common locations depending on current working directory
    os.path.join(_repo_dir, "models", "plant_disease_model.pth"),
    os.path.join(_app_dir, "models", "plant_disease_model.pth"),
    "models/plant_disease_model.pth",
]
disease_model_candidates = [p for p in disease_model_candidates if p]
disease_model_path = next((p for p in disease_model_candidates if os.path.exists(p)), None)

if disease_model_path:
    try:
        import torch
        from torchvision import transforms
        from utils.model import ResNet9

        disease_model = ResNet9(3, len(disease_classes))
        disease_model.load_state_dict(
            torch.load(disease_model_path, map_location=torch.device('cpu'))
        )
        disease_model.eval()
    except Exception:
        disease_model = None
else:
    disease_model = None

# crop recommendation model (trained with RandomForestClassifier)
# NOTE: The training script saves the model to `app/models/crop_model.pkl`
crop_model = None
crop_model_path = os.path.join('app', 'models', 'crop_model.pkl')
if os.path.exists(crop_model_path):
    try:
        crop_model = joblib.load(crop_model_path)
    except Exception:
        crop_model = None
else:
    crop_model = None

# human‑readable crop names and categories for the recommendation output
CROP_INFO = {
    # Cereals
    'wheat': ('Wheat', 'Cereal'),
    'rice': ('Rice', 'Cereal'),
    'maize': ('Maize', 'Cereal'),
    'barley': ('Barley', 'Cereal'),
    'millet': ('Millet', 'Cereal'),
    # Pulses
    'chickpea': ('Chickpea', 'Pulse'),
    'lentil': ('Lentil', 'Pulse'),
    'pigeon_pea': ('Pigeon Pea', 'Pulse'),
    'green_gram': ('Green Gram (Moong)', 'Pulse'),
    'black_gram': ('Black Gram (Urad)', 'Pulse'),
    # Vegetables
    'tomato': ('Tomato', 'Vegetable'),
    'potato': ('Potato', 'Vegetable'),
    'onion': ('Onion', 'Vegetable'),
    'cabbage': ('Cabbage', 'Vegetable'),
    'cauliflower': ('Cauliflower', 'Vegetable'),
    'spinach': ('Spinach', 'Vegetable'),
    'carrot': ('Carrot', 'Vegetable'),
    'brinjal': ('Brinjal (Eggplant)', 'Vegetable'),
    'capsicum': ('Capsicum', 'Vegetable'),
    # Fruits
    'mango': ('Mango', 'Fruit'),
    'banana': ('Banana', 'Fruit'),
    'apple': ('Apple', 'Fruit'),
    'orange': ('Orange', 'Fruit'),
    'papaya': ('Papaya', 'Fruit'),
    'pomegranate': ('Pomegranate', 'Fruit'),
    'grapes': ('Grapes', 'Fruit'),
    'watermelon': ('Watermelon', 'Fruit'),
    'muskmelon': ('Muskmelon', 'Fruit'),
    # Oil crops
    'mustard': ('Mustard', 'Oil Crop'),
    'groundnut': ('Groundnut', 'Oil Crop'),
    'soybean': ('Soybean', 'Oil Crop'),
    'sunflower': ('Sunflower', 'Oil Crop'),
}

# simple growing tips per crop (can be expanded with more detailed agronomy data)
CROP_TIPS = {
    'wheat': "Prefers cool, dry climate with well‑drained loamy soil. Avoid waterlogging during grain filling.",
    'rice': "Needs warm temperatures and plenty of water; thrives in clayey or loamy soils with good puddling.",
    'maize': "Grows well in warm conditions with moderate rainfall and fertile, well‑drained soil.",
    'barley': "Tolerates cool, dry climates; grows best in well‑drained light to medium soils.",
    'millet': "Highly drought tolerant; suitable for light, sandy to loamy soils in low‑rainfall areas.",
    'chickpea': "Requires cool, dry weather and well‑drained loamy soil with neutral to slightly alkaline pH.",
    'lentil': "Prefers cool climate and moderately fertile, well‑drained loam; avoid heavy waterlogging.",
    'pigeon_pea': "Suited to warm, semi‑arid climates and deep, well‑drained loamy soils.",
    'green_gram': "Grows well in warm, humid climates on light loamy soils with good drainage.",
    'black_gram': "Performs best in warm, humid conditions and fertile, well‑drained soils.",
    'tomato': "Needs warm days, cool nights, and fertile, well‑drained soil rich in organic matter.",
    'potato': "Prefers cool climate and loose, well‑drained sandy loam; avoid high temperatures at tuberization.",
    'onion': "Requires cool weather for vegetative growth and warm, dry conditions for bulb development.",
    'cabbage': "Thrives in cool climate with fertile, well‑drained soil rich in organic matter.",
    'cauliflower': "Needs cool, moist climate and fertile, well‑drained loamy soil with good moisture.",
    'spinach': "Grows best in cool weather on fertile, well‑drained loam rich in nitrogen.",
    'carrot': "Prefers deep, loose, sandy loam soils free of stones for straight root development.",
    'brinjal': "Requires warm climate and fertile, well‑drained soil; sensitive to frost.",
    'capsicum': "Does well in cool to moderately warm climate and rich, well‑drained soils.",
    'mango': "Grows in tropical to subtropical climates on deep, well‑drained loamy soils.",
    'banana': "Needs warm, humid climate and rich, well‑drained soils with assured irrigation.",
    'apple': "Thrives in cool to temperate climates with well‑drained loamy soils and chilling hours.",
    'orange': "Prefers subtropical climate and well‑drained sandy loam; sensitive to waterlogging.",
    'papaya': "Requires warm climate and well‑drained, fertile soil; sensitive to frost and water stagnation.",
    'pomegranate': "Tolerates semi‑arid climate and light, well‑drained soils with low to moderate rainfall.",
    'grapes': "Needs warm, dry climate and deep, well‑drained soils; avoid heavy clay with poor drainage.",
    'watermelon': "Requires hot, dry climate and sandy loam soils with good drainage.",
    'muskmelon': "Grows well in warm, dry climates on light, well‑drained sandy loam.",
    'mustard': "Prefers cool season and fertile, well‑drained loam; sensitive to high temperatures at flowering.",
    'groundnut': "Needs warm climate and loose, sandy loam soils for proper pod development.",
    'soybean': "Grows well in warm, humid climates with fertile, well‑drained loams rich in organic matter.",
    'sunflower': "Tolerates semi‑arid climate and performs well on a wide range of well‑drained soils.",
}


def predict_image(img, model=None):
    if model is None:
        model = disease_model
    if model is None:
        return None

    import torch
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
        ]
    )
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    return prediction


def _get_first(d: dict, *keys: str, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def _parse_crop_inputs(soil_data: dict, weather_data: dict):
    """
    Parses JSON payloads into numeric values expected by the crop model.

    Expected keys:
    - soil_data: nitrogen, phosphorous, pottasium (or potassium), ph
    - weather_data: temperature, humidity, rainfall
    """
    nitrogen = float(_get_first(soil_data, "nitrogen", default=0.0))
    phosphorous = float(_get_first(soil_data, "phosphorous", default=0.0))
    pottasium = float(_get_first(soil_data, "pottasium", "potassium", default=0.0))
    ph_val = float(_get_first(soil_data, "ph", "pH", default=0.0))

    temperature = float(_get_first(weather_data, "temperature", default=0.0))
    humidity = float(_get_first(weather_data, "humidity", default=0.0))
    rainfall = float(_get_first(weather_data, "rainfall", default=0.0))

    return nitrogen, phosphorous, pottasium, temperature, humidity, ph_val, rainfall


def get_top3_predicted_crops(soil_data: dict, weather_data: dict):
    """
    Returns top-3 predicted crops as a list of dicts:
    - crop: crop label (e.g. "maize")
    - probability: float in [0,1] (used by existing UI templates)
    """
    N, P, K, temperature, humidity, ph_val, rainfall = _parse_crop_inputs(soil_data, weather_data)
    input_vector = [[N, P, K, temperature, humidity, ph_val, rainfall]]

    # If the ML model is available, use it.
    if crop_model is not None and hasattr(crop_model, "predict_proba"):
        try:
            proba = crop_model.predict_proba(input_vector)[0]
            classes = list(crop_model.classes_)
            class_probs = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)
            return [{"crop": label, "probability": float(p)} for label, p in class_probs[:3]]
        except Exception:
            # Fall back to heuristic if model inference fails.
            pass

    # Heuristic fallback so the app still works without trained models.
    return heuristic_top3_crops(soil_data, weather_data)


app = Flask(__name__)

_basedir = os.path.abspath(os.path.dirname(__file__))
_instance = os.path.join(_basedir, "instance")
os.makedirs(_instance, exist_ok=True)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-agrozoon-change-me-in-production")
# Default: SQLite in app/instance. Override with DATABASE_URL for MySQL, e.g.
# mysql+pymysql://user:pass@localhost/agrozoon
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL",
    "sqlite:///" + os.path.join(_instance, "agrozoon.db").replace("\\", "/"),
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

db.init_app(app)
app.register_blueprint(auth_bp)

with app.app_context():
    db.create_all()


@app.context_processor
def inject_current_user():
    uid = session.get("user_id")
    user = User.query.get(uid) if uid else None
    return {"current_user": user}


@app.before_request
def require_login_for_app():
    """
    Make the web app accessible only after login.
    Allow: static files + login/signup/logout endpoints.
    """
    endpoint = request.endpoint or ""
    if endpoint.startswith("static"):
        return None

    # Allow auth routes
    if endpoint in {"auth.login", "auth.signup", "auth.logout"}:
        return None

    # Allow favicon if requested
    if request.path in {"/favicon.ico"}:
        return None

    uid = session.get("user_id")
    if not uid:
        return redirect(url_for("auth.login"))
    # If user id is invalid, clear and force login
    if not User.query.get(uid):
        session.clear()
        return redirect(url_for("auth.login"))
    return None


@app.route("/api/crop-library", methods=["GET"])
def api_crop_library():
    """
    Returns crop condition ranges (min/max) for N/P/K, temperature, humidity, pH, rainfall
    based on `app/Data/crop_recommendation_extended.csv` (fallback to `crop_recommendation.csv`).
    """
    import pandas as pd

    base_dir = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.join(base_dir, "Data", "crop_recommendation_extended.csv"),
        os.path.join(base_dir, "Data", "crop_recommendation.csv"),
    ]
    csv_path = next((p for p in candidates if os.path.exists(p)), None)
    if not csv_path:
        return jsonify({"error": "Crop dataset not found."}), 500

    df = pd.read_csv(csv_path)
    needed = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]
    for c in needed:
        if c not in df.columns:
            return jsonify({"error": f"Invalid crop dataset format (missing {c})."}), 500

    out = []
    grp = df.groupby("label", dropna=True)
    for label, g in grp:
        label = str(label).strip()
        if not label:
            continue
        display_name, category = CROP_INFO.get(label, (label.replace("_", " ").title(), "Other"))
        tip = CROP_TIPS.get(label)
        row = {
            "label": label,
            "name": display_name,
            "category": category,
            "description": tip or "",
            "conditions": {},
        }
        for k in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]:
            try:
                mn = float(g[k].min())
                mx = float(g[k].max())
                row["conditions"][k] = {"min": mn, "max": mx}
            except Exception:
                row["conditions"][k] = None
        out.append(row)

    out.sort(key=lambda r: (r.get("category") or "", r.get("name") or ""))
    return jsonify({"source": os.path.basename(csv_path), "crops": out})


@app.route('/')
def home():
    title = 'AgroZoon - Home'
    return render_template('index.html', title=title)


@app.route('/dashboard')
def dashboard():
    uid = session.get("user_id")
    if not uid:
        return redirect(url_for("auth.login"))
    user = User.query.get(uid)
    if not user:
        session.clear()
        return redirect(url_for("auth.login"))
    # Keep /dashboard for compatibility, but use the main AgroZoon app as the post-login landing.
    return redirect(url_for("home"))


@app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'AgroZoon - Fertilizer Suggestion'
    # optional crop passed from crop recommendation page (?crop=rice)
    preselected_crop = request.args.get('crop')
    return render_template('fertilizer.html', title=title, preselected_crop=preselected_crop)


@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'AgroZoon - Fertilizer Suggestion'

    crop_name = str(request.form['cropname']).strip()
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])

    df = pd.read_csv('app/Data/fertilizer.csv')

    # ensure crop exists in dataset
    if crop_name not in df['Crop'].values:
        return render_template(
            'fertilizer.html',
            title=title,
            error=f"No fertilizer data available for crop '{crop_name}'. Please choose another crop.",
            preselected_crop=None,
        )

    row = df[df['Crop'] == crop_name].iloc[0]
    nr = int(row['N'])
    pr = int(row['P'])
    kr = int(row['K'])

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'AgroZoon - Disease Detection'

    if disease_model is None:
        searched_paths = ", ".join(disease_model_candidates)
        return render_template(
            'disease.html',
            title=title,
            error=(
                'Disease detection requires plant_disease_model.pth. '
                'Train it using the plant-disease notebook with the New Plant Diseases '
                f'dataset from Kaggle. Checked paths: {searched_paths}'
            ),
        )

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()
            prediction = predict_image(img)
            if prediction is None:
                return render_template('disease.html', title=title)
            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except Exception:
            pass
    return render_template('disease.html', title=title)


# ---- Crop recommendation endpoints ------------------------------------------------
@app.route('/crop')
def crop_recommendation():
    """Render form for entering soil/climate parameters."""
    print("crop_recommendation called")
    title = 'AgroZoon - Crop Recommendation'
    return render_template('crop.html', title=title)


@app.route('/predict-crop', methods=['POST'])
def predict_crop():
    """Take form values, run the trained classifier, and show results."""
    title = 'AgroZoon - Crop Recommendation'

    # validate inputs
    try:
        N = float(request.form['nitrogen'])
        P = float(request.form['phosphorous'])
        K = float(request.form['pottasium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph_val = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
    except Exception:
        return render_template('crop.html', title=title, error="Please enter valid numeric values for all fields.")

    if crop_model is None:
        return render_template(
            'crop.html',
            title=title,
            error="Crop recommendation model not loaded. Please train the model first.",
        )

    input_vector = [[N, P, K, temperature, humidity, ph_val, rainfall]]
    try:
        # main prediction
        predicted_label = crop_model.predict(input_vector)[0]

        top_crops = None
        confidence = None
        if hasattr(crop_model, 'predict_proba'):
            proba = crop_model.predict_proba(input_vector)[0]
            classes = list(crop_model.classes_)
            # sort classes by probability, descending
            class_probs = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)
            top_crops = []
            for label, p in class_probs[:3]:
                display_name, category = CROP_INFO.get(label, (label, 'Other'))
                top_crops.append(
                    {
                        'label': label,
                        'name': display_name,
                        'category': category,
                        'probability': p,
                    }
                )
            # confidence of the best prediction
            confidence = class_probs[0][1]

        # map best label to human‑readable name and category
        display_name, category = CROP_INFO.get(predicted_label, (predicted_label, 'Other'))
        tip = CROP_TIPS.get(predicted_label)

        return render_template(
            'crop-result.html',
            crop=display_name,
            crop_label=predicted_label,
            confidence=confidence,
            category=category,
            top_crops=top_crops,
            tip=tip,
            title=title,
        )
    except Exception as e:
        return render_template('crop.html', title=title, error=str(e))


# --------------------------- Profit Optimizer UI + API ---------------------------
@app.route('/profit-optimizer-ui')
def profit_optimizer_ui():
    title = 'AgroZoon - Profit Optimizer'
    return render_template('profit-optimizer.html', title=title)


@app.route('/predict-profit', methods=['POST'])
def predict_profit():
    def get_required_float(key: str) -> float:
        raw = request.form.get(key, None)
        if raw is None or str(raw).strip() == "":
            raise ValueError(f"Missing field: {key}")
        return float(raw)

    def get_optional_float(key: str, default: float) -> float:
        raw = request.form.get(key, None)
        if raw is None or str(raw).strip() == "":
            return default
        return float(raw)

    def to_bool(v: object) -> bool:
        s = str(v).strip().lower()
        return s in {"1", "true", "yes", "on"}

    crop_type = str(request.form.get("crop_type", "")).strip().lower()
    if crop_type not in CROP_DISPLAY:
        return jsonify({"error": "Invalid crop type."}), 400

    # Core inputs (required)
    try:
        land_area_acres = get_required_float("land_area_acres")
        cost_seeds = get_required_float("cost_seeds")
        fertilizer_cost = get_required_float("fertilizer_cost")
        labor_cost = get_required_float("labor_cost")
        irrigation_cost = get_required_float("irrigation_cost")
        expected_yield = get_required_float("expected_yield")
        market_price = get_required_float("market_price")
    except Exception as e:
        return jsonify({"error": f"Invalid or missing numeric inputs: {e}"}), 400

    # Optional environmental inputs (for AI mode)
    rainfall_mm = get_optional_float("rainfall_mm", default=500.0)
    temperature_c = get_optional_float("temperature_c", default=25.0)

    use_ai = to_bool(request.form.get("use_ai", "0"))

    # Optional comparison inputs
    compare_enabled = to_bool(request.form.get("compare_crop_2", "0"))
    crop_type_2_raw = str(request.form.get("crop_type_2", "")).strip().lower()
    crop_type_2 = crop_type_2_raw if compare_enabled and crop_type_2_raw in CROP_DISPLAY else None

    expected_yield_2 = None
    market_price_2 = None
    if crop_type_2:
        if not use_ai:
            try:
                expected_yield_2 = get_required_float("expected_yield_2")
                market_price_2 = get_required_float("market_price_2")
            except Exception as e:
                return jsonify({"error": f"Second crop comparison requires expected_yield_2 and market_price_2: {e}"}), 400
        else:
            # In AI mode, the backend will predict yield/price for crop 2.
            expected_yield_2 = get_optional_float("expected_yield_2", default=0.0)
            market_price_2 = get_optional_float("market_price_2", default=0.0)

    # Resolve expected yield & market price used for calculation.
    if use_ai:
        expected_yield_used, market_price_used = predict_yield_and_future_price_ml(crop_type, rainfall_mm, temperature_c)
    else:
        expected_yield_used, market_price_used = expected_yield, market_price

    profit_1 = calc_profit(
        land_area_acres=land_area_acres,
        expected_yield_kg_per_acre=expected_yield_used,
        market_price_inr_per_kg=market_price_used,
        cost_seeds=cost_seeds,
        cost_fertilizer=fertilizer_cost,
        cost_labor=labor_cost,
        cost_irrigation=irrigation_cost,
    )

    compare_obj = None
    # If crop2 is enabled, resolve its used values too.
    if crop_type_2:
        if use_ai:
            expected_yield_2_used, market_price_2_used = predict_yield_and_future_price_ml(
                crop_type_2, rainfall_mm, temperature_c
            )
        else:
            expected_yield_2_used, market_price_2_used = expected_yield_2, market_price_2

        profit_2 = calc_profit(
            land_area_acres=land_area_acres,
            expected_yield_kg_per_acre=expected_yield_2_used,
            market_price_inr_per_kg=market_price_2_used,
            cost_seeds=cost_seeds,
            cost_fertilizer=fertilizer_cost,
            cost_labor=labor_cost,
            cost_irrigation=irrigation_cost,
        )
        compare_obj = {
            "crop_type_2": crop_type_2,
            "total_cost": profit_2["total_cost"],
            "revenue": profit_2["revenue"],
            "profit": profit_2["profit"],
        }

    # Suggest a better crop.
    # Note: the suggestion function recomputes crop-1 profits based on `use_ai` so we pass the same
    # used yield/price values for consistency.
    suggestion, _ = suggest_crop_for_better_profit(
        crop_1=crop_type,
        crop_2=crop_type_2,
        land_area_acres=land_area_acres,
        cost_seeds=cost_seeds,
        cost_fertilizer=fertilizer_cost,
        cost_labor=labor_cost,
        cost_irrigation=irrigation_cost,
        expected_yield_1=expected_yield_used,
        market_price_1=market_price_used,
        expected_yield_2=expected_yield_2 if expected_yield_2 is not None else (expected_yield_2_used if crop_type_2 else None),
        market_price_2=market_price_2 if market_price_2 is not None else (market_price_2_used if crop_type_2 else None),
        rainfall_mm=rainfall_mm,
        temperature_c=temperature_c,
        use_ai=use_ai,
    )

    response = {
        "total_cost": profit_1["total_cost"],
        "revenue": profit_1["revenue"],
        "profit": profit_1["profit"],
        "suggestion": suggestion,
    }
    if compare_obj:
        response["compare"] = compare_obj

    return jsonify(response)


# --------------------------- JSON APIs for the Dashboard ---------------------------


@app.route('/api/crop-recommendation', methods=['POST'])
def api_crop_recommendation():
    payload = request.get_json(silent=True) or {}
    soil_data = payload.get("soil_data") or {}
    weather_data = payload.get("weather_data") or {}

    try:
        top_pred = get_top3_predicted_crops(soil_data, weather_data)
        prob_map = {item["crop"]: float(item.get("probability", 0.0)) for item in top_pred}

        profit_rows, best_crop_label, _best_profit, _sat = rank_profit_table(
            top_pred, soil_data, weather_data
        )
        best_display, _best_category = CROP_INFO.get(best_crop_label, (best_crop_label, "Other"))
        why = why_this_crop_is_profitable(best_crop_label, profit_rows, soil_data, weather_data)

        top_crops = []
        for row in profit_rows:
            label = row["crop"]
            display_name, category = CROP_INFO.get(label, (label, "Other"))
            top_crops.append(
                {
                    "crop": label,
                    "name": display_name,
                    "category": category,
                    "probability": prob_map.get(label, 0.0),
                    "profit": int(round(row.get("profit", 0.0))),
                }
            )

        return jsonify(
            {
                "best_crop": best_display,
                "best_crop_label": best_crop_label,
                "top_crops": top_crops,
                "why_recommended": why,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/fertilizer-optimizer', methods=['POST'])
def api_fertilizer_optimizer():
    payload = request.get_json(silent=True) or {}
    crop_name = str(_get_first(payload, "cropname", "crop_name", default="")).strip()
    try:
        N = int(float(_get_first(payload, "nitrogen", default=0)))
        P = int(float(_get_first(payload, "phosphorous", default=0)))
        K = int(float(_get_first(payload, "pottasium", "potassium", default=0)))
    except Exception:
        return jsonify({"error": "Invalid nutrient inputs."}), 400

    if not crop_name:
        return jsonify({"error": "Missing crop name."}), 400

    df = pd.read_csv('app/Data/fertilizer.csv')

    if crop_name not in df['Crop'].values:
        return jsonify({"error": f"No fertilizer data available for crop '{crop_name}'."}), 400

    row = df[df['Crop'] == crop_name].iloc[0]
    nr = int(row['N'])
    pr = int(row['P'])
    kr = int(row['K'])

    n = nr - N
    p = pr - P
    k = kr - K

    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        key = "NHigh" if n < 0 else "Nlow"
    elif max_value == "P":
        key = "PHigh" if p < 0 else "Plow"
    else:
        key = "KHigh" if k < 0 else "Klow"

    recommendation = Markup(str(fertilizer_dic[key]))
    return jsonify({"recommendation_html": str(recommendation)})


@app.route('/api/disease-detect', methods=['POST'])
def api_disease_detect():
    if disease_model is None:
        return (
            jsonify(
                {
                    "error": (
                        "Disease detection requires plant_disease_model.pth. "
                        "Train it using the plant-disease notebook with the New Plant Diseases dataset from Kaggle."
                    )
                }
            ),
            400,
        )

    if 'file' not in request.files:
        return jsonify({"error": "Missing file upload. Use form-data key 'file'."}), 400

    file = request.files.get('file')
    if not file:
        return jsonify({"error": "Empty upload."}), 400

    try:
        img = file.read()
        prediction = predict_image(img)
        if prediction is None:
            return jsonify({"error": "Could not process image."}), 400
        prediction_html = Markup(str(disease_dic[prediction]))
        return jsonify({"prediction": prediction, "prediction_html": str(prediction_html)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/profit-optimizer', methods=['POST'])
def profit_optimizer():
    payload = request.get_json(silent=True) or {}
    soil_data = payload.get("soil_data") or {}
    weather_data = payload.get("weather_data") or {}

    try:
        top_pred = get_top3_predicted_crops(soil_data, weather_data)
        profit_rows, best_crop_label, _best_profit, _sat = rank_profit_table(
            top_pred, soil_data, weather_data
        )

        best_display, _best_cat = CROP_INFO.get(best_crop_label, (best_crop_label, "Other"))
        why = why_this_crop_is_profitable(best_crop_label, profit_rows, soil_data, weather_data)

        profit_table = []
        for row in profit_rows:
            crop_label = row["crop"]
            crop_display, _cat = CROP_INFO.get(crop_label, (crop_label, "Other"))
            profit_table.append({"crop": crop_display, "profit": int(round(row.get("profit", 0.0)))})

        # Required fields + bonus explanation.
        return jsonify(
            {
                "recommended_crop": best_display,
                "profit_table": profit_table,
                "why_recommended": why,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    # show available routes for debugging
    print("Registered routes:")
    for rule in app.url_map.iter_rules():
        print(rule)
    app.run(debug=True)

