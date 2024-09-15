from . import db
from flask_login import UserMixin
from sqlalchemy import func


class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    transaction_type = db.Column(db.String(50))
    amount = db.Column(db.Float)
    nameOrig = db.Column(db.String(100))
    oldBalanceOrig = db.Column(db.Float)
    newBalanceOrig = db.Column(db.Float)
    nameDest = db.Column(db.String(100))
    oldBalanceDest = db.Column(db.Float)
    newBalanceDest = db.Column(db.Float)
    label = db.Column(db.Boolean)
    date = db.Column(db.DateTime(timezone=True), default=func.now())
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    username = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    transactions = db.relationship("Transaction")
