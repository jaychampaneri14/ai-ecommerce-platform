"""
AI E-Commerce Platform — Flask + SQLite + ML recommendations
"""
from flask import Flask, render_template_string, request, jsonify, session
import sqlite3, json, hashlib, os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize DB
def init_db():
    conn = sqlite3.connect("shop.db")
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT, category TEXT, price REAL,
            description TEXT, stock INTEGER, rating REAL
        );
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE, password TEXT
        );
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY, user_id INTEGER,
            product_id INTEGER, quantity INTEGER, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS cart (
            id INTEGER PRIMARY KEY, user_id INTEGER,
            product_id INTEGER, quantity INTEGER
        );
    """)
    # Seed products
    products = [
        ("Laptop Pro 15", "Electronics", 1299.99, "High performance laptop with AI capabilities", 50, 4.5),
        ("Wireless Headphones", "Electronics", 149.99, "Noise-canceling Bluetooth headphones", 200, 4.7),
        ("Smart Watch", "Electronics", 299.99, "Health tracking smartwatch with GPS", 75, 4.3),
        ("Running Shoes", "Sports", 89.99, "Lightweight performance running shoes", 150, 4.6),
        ("Yoga Mat", "Sports", 35.99, "Non-slip eco-friendly yoga mat", 300, 4.8),
        ("Coffee Maker", "Kitchen", 79.99, "Programmable 12-cup coffee maker", 100, 4.4),
        ("Air Purifier", "Home", 199.99, "HEPA filter air purifier for large rooms", 60, 4.5),
        ("Desk Lamp", "Home", 45.99, "LED desk lamp with adjustable brightness", 180, 4.3),
        ("Python Programming Book", "Books", 39.99, "Complete guide to Python and AI/ML", 500, 4.9),
        ("Backpack", "Accessories", 59.99, "Waterproof laptop backpack 30L", 250, 4.6),
    ]
    c.execute("SELECT COUNT(*) FROM products")
    if c.fetchone()[0] == 0:
        c.executemany("INSERT INTO products (name, category, price, description, stock, rating) VALUES (?,?,?,?,?,?)", products)
    conn.commit()
    conn.close()

def get_db():
    conn = sqlite3.connect("shop.db")
    conn.row_factory = sqlite3.Row
    return conn

# ML Recommendation Engine
class RecommendationEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100)
        self._fit()

    def _fit(self):
        conn = get_db()
        products = conn.execute("SELECT id, name, category, description FROM products").fetchall()
        conn.close()
        self.products = [dict(p) for p in products]
        texts = [f"{p['name']} {p['category']} {p['description']}" for p in self.products]
        if texts:
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def get_similar(self, product_id, n=4):
        ids = [p["id"] for p in self.products]
        if product_id not in ids:
            return []
        idx = ids.index(product_id)
        sims = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        sims[idx] = 0  # exclude self
        top_idx = sims.argsort()[-n:][::-1]
        return [self.products[i] for i in top_idx]

    def get_personalized(self, user_id, n=5):
        conn = get_db()
        orders = conn.execute(
            "SELECT product_id FROM orders WHERE user_id=? ORDER BY created_at DESC LIMIT 5",
            (user_id,)).fetchall()
        conn.close()
        if not orders:
            return self.products[:n]
        purchased_ids = [o["product_id"] for o in orders]
        recs = []
        for pid in purchased_ids:
            recs.extend(self.get_similar(pid, n=3))
        seen = set(purchased_ids)
        unique = []
        for p in recs:
            if p["id"] not in seen:
                seen.add(p["id"])
                unique.append(p)
        return unique[:n] if unique else self.products[:n]

engine = RecommendationEngine()

HTML = """<!DOCTYPE html>
<html>
<head>
<title>AI Shop</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: Arial, sans-serif; background: #f5f5f5; }
nav { background: #1a1a2e; color: white; padding: 15px 20px; display: flex; align-items: center; gap: 20px; }
nav h1 { font-size: 1.4em; flex: 1; }
nav a { color: #a8dadc; text-decoration: none; padding: 8px 15px; border-radius: 4px; background: rgba(255,255,255,0.1); }
.container { max-width: 1200px; margin: 20px auto; padding: 0 20px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 20px; }
.card { background: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
.card h3 { margin-bottom: 8px; }
.price { color: #e63946; font-size: 1.2em; font-weight: bold; margin: 8px 0; }
.category { background: #a8dadc; padding: 3px 8px; border-radius: 12px; font-size: 0.8em; display: inline-block; }
.rating { color: #fca311; margin: 5px 0; }
.btn { background: #1a1a2e; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; width: 100%; margin-top: 8px; }
.btn:hover { background: #16213e; }
.search { width: 100%; padding: 10px 15px; border: 1px solid #ddd; border-radius: 4px; margin-bottom: 20px; font-size: 1em; }
h2 { margin: 20px 0 10px; }
</style>
</head>
<body>
<nav>
  <h1>🛍️ AI Shop</h1>
  <a href="/">Home</a>
  <a href="/cart">Cart 🛒</a>
</nav>
<div class="container">
  <input class="search" id="search" placeholder="Search products..." oninput="searchProducts(this.value)">
  <h2>All Products</h2>
  <div class="grid" id="products"></div>
  <h2>Recommended For You</h2>
  <div class="grid" id="recs"></div>
</div>
<script>
let allProducts = [];
async function load() {
  const r = await fetch("/api/products");
  allProducts = await r.json();
  render(allProducts, "products");
  const rr = await fetch("/api/recommendations");
  render(await rr.json(), "recs");
}
function render(products, elId) {
  const el = document.getElementById(elId);
  el.innerHTML = products.map(p => `
    <div class="card">
      <span class="category">${p.category}</span>
      <h3>${p.name}</h3>
      <p style="font-size:0.85em;color:#666">${p.description}</p>
      <div class="price">$${p.price}</div>
      <div class="rating">${"★".repeat(Math.round(p.rating))} (${p.rating})</div>
      <button class="btn" onclick="addCart(${p.id})">Add to Cart</button>
    </div>`).join("");
}
function searchProducts(q) {
  const filtered = q ? allProducts.filter(p =>
    (p.name + p.description + p.category).toLowerCase().includes(q.toLowerCase())) : allProducts;
  render(filtered, "products");
}
async function addCart(id) {
  const r = await fetch("/api/cart/add", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({product_id:id,quantity:1})});
  const d = await r.json();
  alert(d.message || "Added to cart!");
}
load();
</script>
</body></html>"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api/products")
def api_products():
    conn = get_db()
    q = request.args.get("q", "")
    if q:
        products = conn.execute(
            "SELECT * FROM products WHERE name LIKE ? OR category LIKE ? OR description LIKE ?",
            (f"%{q}%", f"%{q}%", f"%{q}%")).fetchall()
    else:
        products = conn.execute("SELECT * FROM products").fetchall()
    conn.close()
    return jsonify([dict(p) for p in products])

@app.route("/api/recommendations")
def api_recommendations():
    user_id = session.get("user_id", 1)
    recs = engine.get_personalized(user_id, n=4)
    if not recs:
        conn = get_db()
        recs = [dict(p) for p in conn.execute(
            "SELECT * FROM products ORDER BY rating DESC LIMIT 4").fetchall()]
        conn.close()
    return jsonify([dict(p) for p in recs])

@app.route("/api/cart/add", methods=["POST"])
def add_to_cart():
    data = request.json
    product_id = data.get("product_id")
    quantity = data.get("quantity", 1)
    user_id = session.get("user_id", 1)
    conn = get_db()
    existing = conn.execute(
        "SELECT * FROM cart WHERE user_id=? AND product_id=?", (user_id, product_id)).fetchone()
    if existing:
        conn.execute("UPDATE cart SET quantity=quantity+? WHERE id=?", (quantity, existing["id"]))
    else:
        conn.execute("INSERT INTO cart (user_id, product_id, quantity) VALUES (?,?,?)",
                     (user_id, product_id, quantity))
    conn.commit()
    product = conn.execute("SELECT name FROM products WHERE id=?", (product_id,)).fetchone()
    conn.close()
    name = product["name"] if product else "Item"
    return jsonify({"message": f"Added {name} to cart!", "ok": True})

@app.route("/api/cart")
def get_cart():
    user_id = session.get("user_id", 1)
    conn = get_db()
    items = conn.execute("""
        SELECT p.name, p.price, c.quantity, (p.price * c.quantity) as total
        FROM cart c JOIN products p ON p.id = c.product_id
        WHERE c.user_id=?""", (user_id,)).fetchall()
    conn.close()
    return jsonify([dict(i) for i in items])

@app.route("/cart")
def cart_page():
    return render_template_string("""<html><head><title>Cart</title></head><body>
    <h1>Your Cart</h1><div id="items"></div>
    <script>fetch("/api/cart").then(r=>r.json()).then(d=>{
    document.getElementById("items").innerHTML = d.map(i=>
    `<p>${i.name} x${i.quantity} = $${i.total.toFixed(2)}</p>`).join("") || "<p>Empty cart</p>";
    });</script><a href="/">← Continue Shopping</a></body></html>""")

if __name__ == "__main__":
    init_db()
    print("AI E-Commerce Platform running on http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
