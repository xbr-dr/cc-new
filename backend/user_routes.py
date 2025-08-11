from flask import Blueprint, request, jsonify
from rag_generator import generate_answer
from admin_routes import LOCATIONS  # import shared location data

user_bp = Blueprint("user", __name__, url_prefix="/user")

@user_bp.route("/locations", methods=["GET"])
def get_locations():
    return jsonify(LOCATIONS)

@user_bp.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    history = data.get("history", [])
    if not history or not isinstance(history, list):
        return jsonify({"reply": "Please send a valid chat history."})

    reply = generate_answer(history)
    return jsonify({"reply": reply})
