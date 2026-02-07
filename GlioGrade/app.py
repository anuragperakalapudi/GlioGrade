from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from datetime import datetime
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import shutil
import atexit
import tensorflow as tf

app = Flask(__name__, template_folder='templates', static_folder='static')

UPLOADS_DIR = 'uploads'
SLICE_DIR = 'static/slices'

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(SLICE_DIR, exist_ok=True)

TYPE_MODEL_PATH = 'models/type_model.keras'
GRADE_MODEL_PATH = 'models/grade_model.keras'
type_model = tf.keras.models.load_model(TYPE_MODEL_PATH)
grade_model = tf.keras.models.load_model(GRADE_MODEL_PATH)

def cleanup_all():
    if os.path.exists(UPLOADS_DIR):
        shutil.rmtree(UPLOADS_DIR)
    if os.path.exists(SLICE_DIR):
        shutil.rmtree(SLICE_DIR)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(SLICE_DIR, exist_ok=True)
atexit.register(cleanup_all)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['scan']
        if not file or not file.filename.lower().endswith(('.nii', '.nii.gz')):
            return render_template('error.html', error_message="Invalid file type. Upload a .nii or .nii.gz file.")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        folder = os.path.join(UPLOADS_DIR, timestamp)
        os.makedirs(folder, exist_ok=True)

        scan_path = os.path.join(folder, 'T1.nii.gz')
        file.save(scan_path)

        img = nib.load(scan_path)
        data = img.get_fdata()
        data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)  

        resized = tf.image.resize(data[:, :, data.shape[2] // 2], (224, 224)) 
        input_tensor = np.expand_dims(np.expand_dims(resized, axis=-1), axis=0)

        type_pred = type_model.predict(input_tensor)[0]
        grade_pred = grade_model.predict(input_tensor)[0]

        tumor_type = ['Glioblastoma, IDH-wildtype', 'Astrocytoma, IDH-wildtype', 'Oligodendroglioma, IDH-mutant, 1p/19q-codeleted', 'Astrocytoma, IDH-mutant'][np.argmax(type_pred)]
        grade = ['2', '3', '4'][np.argmax(grade_pred)]

        confidence_type = round(100 * np.max(type_pred), 2)
        confidence_grade = round(100 * np.max(grade_pred), 2)

        generate_slices(scan_path, timestamp)

        return redirect(url_for('results', folder=timestamp, tumor_type=tumor_type,
                                grade=grade, confidence_type=confidence_type, confidence_grade=confidence_grade))

    return render_template('upload.html')

def generate_slices(nifti_path, folder):
    img = nib.load(nifti_path)
    data = img.get_fdata()
    data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

    orientations = {
        'axial': data,
        'coronal': np.transpose(data, (0, 2, 1)),
        'sagittal': np.transpose(data, (1, 2, 0))
    }

    slice_folder = os.path.join(SLICE_DIR, folder)
    os.makedirs(slice_folder, exist_ok=True)

    for orientation, oriented_data in orientations.items():
        num_slices = oriented_data.shape[2]
        index_range = range(num_slices)

        for i, slice_idx in enumerate(index_range):
            slice_img = oriented_data[:, :, slice_idx]
            if orientation in ['coronal', 'sagittal', 'axial']:
                slice_img = np.rot90(slice_img, k=1)

            path = os.path.join(slice_folder, f"{orientation}_{i}.png")
            plt.imsave(path, slice_img, cmap='gray')

@app.route('/results/<folder>')
def results(folder):
    tumor_type = request.args.get('tumor_type', 'Unknown')
    grade = request.args.get('grade', 'Unknown')
    confidence_type = request.args.get('confidence_type', 0)
    confidence_grade = request.args.get('confidence_grade', 0)
    return render_template('results.html', folder=folder, tumor_type=tumor_type, grade=grade,
                           confidence_type=confidence_type, confidence_grade=confidence_grade)

@app.route('/get_slices/<folder>/<orientation>')
def get_slices(folder, orientation):
    import re
    slice_folder = os.path.join(SLICE_DIR, folder)
    filenames = [
        f for f in os.listdir(slice_folder)
        if f.startswith(f"{orientation}_") and f.endswith(".png")
    ]

    def extract_index(name):
        match = re.search(rf"{orientation}_(\d+)\.png", name)
        return int(match.group(1)) if match else -1

    sorted_paths = sorted(filenames, key=extract_index)
    full_paths = [f"/static/slices/{folder}/{f}" for f in sorted_paths]

    return jsonify({"slices": full_paths})

if __name__ == '__main__':
    app.run(debug=True)
