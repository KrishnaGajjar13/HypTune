from flask import Flask, render_template, request, redirect, url_for, flash
import os, warnings
import torch
import tensorflow as tf
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/"
app.secret_key = "supersecretkey"  # For flashing error messages

# Ensure upload directory exists
UPLOAD_FOLDER = "uploads/"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)



@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        print("Upload function called!")  # Debugging

        if "file" not in request.files:
            flash("No file part", "danger")
            print("No file part in request")  # Debugging
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            flash("No selected file", "danger")
            print("No file selected")  # Debugging
            return redirect(request.url)

        # Save the uploaded file
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)
        print(f"File {file.filename} uploaded successfully!")  # Debugging
        model_path = file
        flash("File uploaded successfully!", "success")

        # Redirect to validation page
        return redirect(url_for("validate_model", model_path=model_path))

    return render_template("index.html")

@app.route("/create_model", methods=["GET", "POST"])
def create_model():
    framework = request.args.get("framework")  # Get framework from URL parameter

    if request.method == "POST":
        model_name = request.form.get("modelName")
        purpose = request.form.get("purpose")

        # Validation
        if len(model_name) > 15:
            flash("Model name should not exceed 15 characters.", "danger")
            return redirect(url_for("create_model", framework=framework))

        if len(purpose) < 100:
            flash("Description should be at least 100 characters long.", "danger")
            return redirect(url_for("create_model", framework=framework))

        flash("Model details submitted successfully!", "success")
        return redirect(url_for("upload_file"))  # Redirect to upload page or another page

    return render_template("create_model.html", framework=framework)

if __name__ == "__main__":
    app.run(debug=True)
