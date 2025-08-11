from flask import Flask
from flask_cors import CORS
from admin_routes import admin_bp
from user_routes import user_bp

app = Flask(__name__)
CORS(app)

# Register Blueprints
app.register_blueprint(user_bp, url_prefix="/user")
app.register_blueprint(admin_bp, url_prefix="/admin")



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    from rag_retriever import load_documents_and_build_index
    load_documents_and_build_index()
    app.run(debug=True)
