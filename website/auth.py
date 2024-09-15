from flask import Blueprint, render_template, request, flash, redirect, url_for
from .models import User
from werkzeug.security import generate_password_hash, check_password_hash
from . import db
from flask_login import login_user, login_required, logout_user, current_user

auth = Blueprint("auth", __name__)


@auth.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        user = User.query.filter_by(username=username).first()
        if user:
            if check_password_hash(user.password, password):
                flash("Logged in successfully!", category="success")
                login_user(user)
                return redirect(url_for("views.home"))
            else:
                flash("Incorrect password, try again!", category="error")
        else:
            flash("User does not exists!", category="error")

    return render_template("login.html", text="Testing", user=current_user)


@auth.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))


@auth.route("/sign-up", methods=["GET", "POST"])
def sign_up():
    if request.method == "POST":
        email = request.form.get("email")
        user_name = request.form.get("username")
        password1 = request.form.get("password1")
        password2 = request.form.get("password2")

        user = User.query.filter_by(username=user_name).first()
        if user:
            flash("User already exists!", category="error")
        elif len(email) < 4:
            flash("Email must be greater than 4 charactors!!", category="error")
        elif password1 != password2:
            flash("Passwords don't match!!", category="error")
        elif len(password1) < 7:
            flash("Password must be at least 7 charactors.", category="error")
        else:
            new_user = User(
                email=email, username=user_name, password=generate_password_hash(password1, method="pbkdf2:sha256")
            )
            db.session.add(new_user)
            db.session.commit()

            flash("User account created...", category="success")

            return redirect(url_for("auth.login"))

    return render_template("sign_up.html", user=current_user)
