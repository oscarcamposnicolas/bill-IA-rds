# app.py (Versión final que usa el motor de inferencia externo)
import os
import uuid

from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

# --- ¡IMPORTAMOS NUESTRO MOTOR DE INFERENCIA! ---
from motor_inferencia import realizar_inferencia

# --- CONFIGURACIÓN DE FLASK ---
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# --- DEFINICIÓN DE LAS PÁGINAS (RUTAS) ---
@app.route("/")
@app.route("/portada")
def portada_page():
    return render_template("portada.html")


@app.route("/inferencia")
def inferencia_page():
    """
    Esta única función ahora maneja tanto la página de inicio ('/')
    como la de inferencia ('/inferencia'), haciendo que 'url_for' la encuentre.
    """
    return render_template("inferencia.html")


@app.route("/fases")
def fases_page():
    return render_template("fases.html")


@app.route("/fases_html/<path:filename>")
def serve_fases_html(filename):
    return send_from_directory("fases_html", filename)


# --- API PARA LA PREDICCIÓN ---
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No se encontró el archivo"}), 400
        file = request.files["file"]
        if file.filename == "" or not allowed_file(file.filename):
            return jsonify({"error": "Archivo no válido"}), 400

        filename = secure_filename(str(uuid.uuid4()) + "_" + file.filename)
        upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(upload_path)

        result_filename = f"result_{filename}"
        result_path_on_disk = os.path.join(STATIC_FOLDER, result_filename)

        # --- LLAMADA A NUESTRO MOTOR DE IA ---
        # Ahora devuelve también el contexto para que lo podamos mostrar
        detecciones, contexto = realizar_inferencia(upload_path, result_path_on_disk)
        # -----------------------------------------------

        result_url = url_for("static", filename=result_filename)

        # Devolvemos la imagen, las detecciones Y el contexto
        return jsonify(
            {"result_image": result_url, "detections": detecciones, "context": contexto}
        )

    except Exception as e:
        print(f"ERROR en /predict: {e}")
        return jsonify({"error": f"Ocurrió un error en el servidor: {e}"}), 500


# --- Iniciar la aplicación ---
if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(STATIC_FOLDER, exist_ok=True)
    app.run(debug=True)
