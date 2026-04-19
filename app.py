from flask import Flask, jsonify, redirect, render_template, request
import json
from dotenv import load_dotenv

from db.init_db import init_db
from utils.event_tracker import add_to_cart, get_cart, track_event

app = Flask(__name__)
load_dotenv()
init_db()

with open("data/products.json") as f:
    products = json.load(f)

@app.route("/")
def home():
    user_id = "user_1"
    return render_template("index.html", products=products)

@app.route("/click/<int:item_id>")
def click(item_id):
    user_id = "user_1"
    track_event(user_id, item_id, "view")
    track_event(user_id, item_id, "click")
    return redirect("/")

@app.route("/add_to_cart/<int:item_id>")
def add_to_cart(item_id):
    user_id = "user_1"
    add_to_cart(user_id, item_id)
    track_event(user_id, item_id, "add_to_cart")
    return redirect("/cart")

@app.route("/cart")
def cart():
    user_id = "user_1"
    rows = get_cart(user_id)
    product_map = {p["id"]: p for p in products}
    items = []
    for row in rows:
        item_id = int(row["item_id"])
        if item_id in product_map:
            items.append(
                {
                    "name": product_map[item_id]["name"],
                    "quantity": int(row["quantity"]),
                    "price": int(product_map[item_id]["price"]),
                }
            )
    return render_template("cart.html", items=items)


@app.route("/time_spent/<int:item_id>", methods=["POST"])
def time_spent(item_id):
    user_id = "user_1"
    payload = request.get_json(silent=True) or {}
    seconds = float(payload.get("seconds", 0.0))
    if seconds > 0:
        track_event(user_id, item_id, "time_spent", seconds)
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)
