from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Orders(db.Model):
    __tablename__ = 'orders'

    id = db.Column(db.Integer, primary_key=True)
    order_number = db.Column(db.String(50))
    supplier_id = db.Column(db.Integer)
    product_code = db.Column(db.String(50))
    quantity = db.Column(db.Integer)
    unit_price = db.Column(db.Float)
    total_amount = db.Column(db.Float)
    order_date = db.Column(db.DateTime, default=datetime.utcnow)
    expected_delivery = db.Column(db.DateTime)
    status = db.Column(db.String(20))
    user_id = db.Column(db.Integer)

class Clients(db.Model):
    __tablename__ = 'clients'

    id = db.Column(db.Integer, primary_key=True)
    client_code = db.Column(db.String(50))
    name = db.Column(db.String(200))

class Products(db.Model):
    __tablename__ = 'products'

    id = db.Column(db.Integer, primary_key=True)
    is_folder = db.Column(db.Boolean, default=False)
    parent_id = db.Column(db.Integer)
    code = db.Column(db.String(50))
    descr = db.Column(db.String(200))
    article = db.Column(db.String(100))
    measure = db.Column(db.String(20))
    nds_rate = db.Column(db.Float)

class Sales(db.Model):
    __tablename__ = 'sales'

    id = db.Column(db.Integer, primary_key=True)
    product_code = db.Column(db.String(50))
    client_code = db.Column(db.String(50))
    earned = db.Column(db.Float)
    value = db.Column(db.Integer)
    date = db.Column(db.DateTime, default=datetime.utcnow)

class StockLevels(db.Model):
    __tablename__ = 'stock_levels'

    id = db.Column(db.Integer, primary_key=True)
    product_code = db.Column(db.String(50))
    value = db.Column(db.Integer)
    date = db.Column(db.DateTime, default=datetime.utcnow)

class Suppliers(db.Model):
    __tablename__ = 'suppliers'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200))
    contact_person = db.Column(db.String(100))
    phone = db.Column(db.String(20))
    email = db.Column(db.String(100))
    address = db.Column(db.Text)

class Users(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80))
    email = db.Column(db.String(120))
    password_hash = db.Column(db.String(200))
    full_name = db.Column(db.String(150))
    role = db.Column(db.String(50))
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

__all__ = [
    'db',
    'Products',
    'Clients',
    'Orders',
    'Sales',
    'StockLevels',
    'Suppliers',
    'Users'
]