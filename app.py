from flask import Flask, request, render_template, send_from_directory
import os
from model import process_excel_pipeline
from model import progress


app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', plot_div=None, download_url=None)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded."
    file = request.files['file']
    if file.filename == '':
        return "No file selected."

    # Save uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    try:
        # Run pipeline and get results
        plot_div, output_filename, stats = process_excel_pipeline(file_path, app.config['UPLOAD_FOLDER'])

        return render_template(
            'index.html',
            plot_div=plot_div,
            download_url=f"/download/{output_filename}",
            stats=stats
        )

    except Exception as e:
        return f"<h3 style='color:red;'>❌ Error:</h3><pre>{e}</pre>"

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/progress')
def get_progress():
    return {"percent": progress["percent"]}



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

