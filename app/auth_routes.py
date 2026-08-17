import bcrypt

from flask import Blueprint, jsonify, redirect, render_template, request, session, url_for

from models import User, db

auth_bp = Blueprint("auth", __name__)


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def check_password(plain: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        if session.get("user_id"):
            return redirect(url_for("home"))
        return render_template("login.html", title="AgroZoon - Login")

    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    remember_me = request.form.get("remember") == "on"

    if not email or not password:
        return render_template("login.html", title="AgroZoon - Login", auth_error="Email and password are required."), 400

    user = User.query.filter_by(email=email).first()
    if not user or not check_password(password, user.password_hash):
        return render_template("login.html", title="AgroZoon - Login", auth_error="Invalid email or password."), 401

    session.clear()
    session["user_id"] = user.id
    session.permanent = remember_me

    return redirect(url_for("home"))


@auth_bp.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        if session.get("user_id"):
            return redirect(url_for("home"))
        return render_template("signup.html", title="AgroZoon - Signup")

    full_name = (request.form.get("full_name") or "").strip()
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    confirm = request.form.get("confirm_password") or ""

    errors = []
    if not full_name:
        errors.append("Full name is required.")
    if not email:
        errors.append("Email is required.")
    if len(password) < 6:
        errors.append("Password must be at least 6 characters.")
    if password != confirm:
        errors.append("Passwords do not match.")

    if errors:
        msg = " ".join(errors)
        return render_template("signup.html", title="AgroZoon - Signup", auth_error=msg), 400

    if "@" not in email or "." not in email.split("@")[-1]:
        msg = "Please enter a valid email address."
        return render_template("signup.html", title="AgroZoon - Signup", auth_error=msg), 400

    if User.query.filter_by(email=email).first():
        msg = "An account with this email already exists."
        return render_template("signup.html", title="AgroZoon - Signup", auth_error=msg), 409

    user = User(full_name=full_name, email=email, password_hash=hash_password(password))
    db.session.add(user)
    db.session.commit()

    session.clear()
    session["user_id"] = user.id
    session.permanent = True

    return redirect(url_for("home"))


@auth_bp.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    # If it's a fetch/XHR, respond with JSON; otherwise go back home.
    if request.method == "POST" and (request.headers.get("X-Requested-With") == "XMLHttpRequest"):
        return jsonify({"ok": True})
    return redirect(url_for("home"))


@auth_bp.route("/api/me", methods=["GET"])
def me():
    uid = session.get("user_id")
    if not uid:
        return jsonify({"error": "Not authenticated."}), 401
    user = User.query.get(uid)
    if not user:
        session.clear()
        return jsonify({"error": "Not authenticated."}), 401
    return jsonify({"user": user.to_public_dict()})
