from src import app
from src.preview_augmentations import get_preview
from flask import render_template, send_from_directory, request, jsonify, make_response, send_from_directory
import cv2
import os
import json
import shutil
import sys
sys.path.append("data/traffic_sign_interiit")
from data.traffic_sign_interiit.dataset import prepare_new_classes
import numpy as np
from data.traffic_sign_interiit import train

app.config["JSON_PATH"] = "data/"
app.config["DATA_PATH"] = "data/traffic_sign_interiit/"
print(sys.path)
orig_classes = []
self_classes = []

# load json files
with open(app.config["JSON_PATH"] + "ORIG_CLASSES.json") as json_file:
    orig_classes = json.load(json_file)["ORIG_CLASSES"]

with open(app.config["JSON_PATH"] + "SELF_CLASSES.json") as json_file:
    self_classes = json.load(json_file)["SELF_CLASSES"]

# main views of the application


@app.route("/", methods=["GET", "POST"])
def root():
    return render_template("index.html")


@app.route("/addimages", methods=["GET", "POST"])
def addImages():
    if request.method == 'POST':
        if request.form['submit_button'] == "Upload":
            try:
                for f in request.files.getlist('file_name'):
                    selectValue = request.form.get('class-dropdown')
                    upload_folder = ""
                    # upload folder  = class id
                    class_id = get_index(orig_classes, selectValue)
                    if class_id != -1:
                        upload_folder = str(class_id)
                    else:
                        upload_folder = str(
                            get_index(self_classes, selectValue)+44)
                    # check if path exists and upload. else create path and upload
                    path = app.config["DATA_PATH"] + \
                        "dataset/New/" + upload_folder + "/"
                    if os.path.exists(path) == False:
                        os.mkdir(path)
                    f.save(os.path.join(path, f.filename))
                # make the EXTRA folder again
                remake_EXTRA_folder()
                return render_template("addimages.html", msg="Images uploaded successfully", self_classes=self_classes, all_classes=orig_classes + self_classes)
            except:
                print("Except. Do nothing")
    return render_template("addimages.html", self_classes=self_classes, all_classes=orig_classes + self_classes)


@app.route("/addimages/receive_self_class", methods=["POST"])
def save_new_class():
    new_class = request.get_json()['new_class']
    self_classes.append(new_class)
    save_self_classes_json()
    return make_response(jsonify({"message": "Class added!"}), 200)


@app.route("/addimages/remove_self_class", methods=["POST"])
def remove_new_class():
    new_class = request.get_json()['delete_class']
    for klass in new_class:
        class_id = get_index(self_classes, klass) + 44
        folder = str(class_id)
        path = os.path.join(app.config["DATA_PATH"] + "dataset/New/", folder)
        if(os.path.exists(path)):
            shutil.rmtree(path)
        class_dirs = os.listdir(app.config["DATA_PATH"] + "dataset/New/")
        for dir in class_dirs:
            if dir == ".gitkeep":
                continue
            if int(dir) > class_id:
                os.rename(os.path.join(app.config["DATA_PATH"] + "dataset/New/", dir), os.path.join(
                    app.config["DATA_PATH"] + "dataset/New/", str(int(dir)-1)))
        self_classes.remove(klass)
    save_self_classes_json()
    # make the EXTRA folder again
    remake_EXTRA_folder()
    return make_response(jsonify({"message": "Class removed!"}), 200)


@app.route("/augmentations", methods=["GET", "POST"])
def addAugmentations():
    return render_template("augmentations.html")


@app.route("/augmentations/preview", methods=["POST"])
def previewAugmentations():
    req = request.get_json()
    image = cv2.imread("src"+req['image'])
    augmented = get_preview([image], req['augmentationList'])
    cv2.imwrite('src/static/images/temps/preview_aug.png', augmented[0])
    return make_response(jsonify({'message': 'Preview saved to static/images/temps/preview_aug.png'}), 200)


@app.route("/augmentations/uploadimage", methods=["POST"])
def uploadAugmentationImage():
    if request.method == 'POST':
        image = request.files['file']
        path = "/static/images/temps/upload_aug.png"
        image.save("src/"+path)
        return make_response(jsonify({'message': 'Uploaded image successfully', 'path': path}), 200)


@app.route("/dataset", methods=["GET", "POST"])
def datasetStatistics():
    return render_template("dataset.html")


@app.route("/training", methods=["GET", "POST"])
def trainModel():
    return render_template("training.html", self_classes=self_classes)


@app.route("/training/train", methods=["POST"])
def ModelTraindata():
    data = request.get_json()
    with open(app.config["DATA_PATH"] + "/config/params.json") as json_file:
        default_configs = json.load(json_file)
    default_configs["experiment"]["model"] = data["model"]
    default_configs["experiment"]["batch_size"] = int(data["batch_size"])
    default_configs["experiment"]["epochs"] = int(data["epochs"])
    default_configs["experiment"]["learning_rate"] = float(data["learning_rate"])
    default_configs["experiment"]["num_classes"] = int(data["num_classes"])
    default_configs["experiment"]["class_ids"] = list(range(0, 43)) + (np.array(data["class_ids"]) + 43).tolist()
    with open(app.config["DATA_PATH"] + "config/temp_config.json", "w") as outfile:
        json.dump(default_configs, outfile)
    remake_EXTRA_folder()
    train.train("temp_config")


    check = True
    ###############

    if check:
        return make_response(jsonify({"message": "Model Trained Successfuly! Check Visualise tab and results tab for more info."}), 200)
    else:
        return make_response(jsonify({"error": "Something is Wrong. Try Again!"}), 400)


@app.route("/visualise", methods=["GET", "POST"])
def visualiseModel():
    return render_template("visualise.html")


@app.route("/netron", methods=["GET", "POST"])
def Model():
    return render_template("netron.html")


@app.route("/results", methods=["GET", "POST"])
def viewResults():
    return render_template("results.html")

# helpers


def save_self_classes_json():
    temp_dict = {}
    temp_dict["SELF_CLASSES"] = self_classes
    with open(app.config["JSON_PATH"] + "SELF_CLASSES.json", "w") as outfile:
        json.dump(temp_dict, outfile)


def get_index(class_list, class_name):
    for i in range(0, len(class_list)):
        if class_list[i] == class_name:
            return i
    return -1


def remake_EXTRA_folder():
    extra_path = os.path.join(app.config["DATA_PATH"], "dataset/EXTRA/")
    new_path = os.path.join(app.config["DATA_PATH"], "dataset/New/")
    shutil.rmtree(extra_path)
    os.mkdir(extra_path)
    prepare_new_classes.prepare_train_val_n_test(new_path, extra_path)
    return
