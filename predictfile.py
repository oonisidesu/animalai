import os
from flask import Flask, request, redirect, url_for, flash, render_template
from werkzeug.utils import secure_filename

from keras.models import Sequential, load_model
import numpy as np
import keras, sys
from PIL import Image

classes = ["monkey","boar", "crow"]
num_classes = len(classes)
image_size = 50

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ファイルのアップロード可否判定関数
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            model = load_model('./animal_cnn_aug.h5')

            image = Image.open(filepath)
            image = image.convert('RGB')
            image = image.resize((image_size, image_size))
            data = np.asarray(image) / 255
            X = []
            X.append(data)
            X = np.array(X)

            result = model.predict([X])[0]
            predicted = result.argmax()
            percentage = int(result[predicted] * 100)

            if classes[predicted] == "boar":
                classes[predicted] = "イノシシ"
            if classes[predicted] == "crow":
                classes[predicted] = "カラス"
            if classes[predicted] == "monkey":
                classes[predicted] = "サル"

            return render_template('kekka.html',
                                    title='機械学習',
                                    image = url_for('uploaded_file', filename=filename),
                                    predicted = classes[predicted],
                                    percentage = str(percentage)
                                    )

            # return redirect(url_for('uploaded_file', filename=filename))
    return render_template('upload.html', title='機械学習')

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)