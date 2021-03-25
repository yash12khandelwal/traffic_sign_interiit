from src import app
from src.preview_augmentations import get_preview
from flask import render_template, send_from_directory, request, jsonify, make_response, send_from_directory, redirect
import cv2
import os
import json
import csv
import random
import numpy
import shutil
import sys
sys.path.append("data/traffic_sign_interiit")
from data.traffic_sign_interiit.dataset import prepare_new_classes
import numpy as np
from data.traffic_sign_interiit import train


app.config["JSON_PATH"] = "data/"
app.config["DATA_PATH"] = "data/traffic_sign_interiit/"
app.config["MEDIA_FOLDER"] = "data/traffic_sign_interiit/dataset/New/"
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

@app.route('/new/<folder>/<path:filename>')
def download_file(folder,filename):
    return send_from_directory(app.config["MEDIA_FOLDER"]+ folder + "/", filename, as_attachment=True)

@app.route("/addimages", methods=["GET", "POST"])
def addImages():
    base_path = app.config["DATA_PATH"] + "dataset/New/"
    img_paths = []
    for i in range(len(orig_classes), len(orig_classes) + len(self_classes)) :
        path = base_path + f'{i}'
        if os.path.exists(path) :
            random_file=random.choice(os.listdir(path))
            temp = [f'{i}', random_file]
            img_paths.append(temp)

    if request.method == 'POST':
        if request.form['submit_button'] == "Upload":
            dataset = request.form['dataset']
            file_list = request.files.getlist('file_name')
            class_name = request.form.get('class-dropdown')
            success = upload_images(dataset, file_list, class_name)
            if success:
                return render_template("addimages.html", msg="Images uploaded successfully", self_classes=self_classes, all_classes=orig_classes + self_classes, img_paths = img_paths)
            else:
                return render_template("addimages.html", msg="Please select images to upload", self_classes=self_classes, all_classes=orig_classes + self_classes, img_paths = img_paths)
    return render_template("addimages.html", self_classes=self_classes, all_classes=orig_classes + self_classes, img_paths = img_paths)


@app.route("/addimages/receive_self_class", methods=["POST"])
def save_new_class():
    new_class = request.get_json()['new_class']
    self_classes.append(new_class)
    save_self_classes_json()
    return make_response(jsonify({"message": "Class added!"}), 200)


@app.route("/addimages/remove_self_class", methods=["POST"])
def remove_new_class():
    class_list = request.get_json()['delete_class']
    delete_classes(class_list)
    return make_response(jsonify({"message": "Class(es) removed!"}), 200)


@app.route("/augmentations", methods=["GET", "POST"])
def addAugmentations():
    return render_template("augmentations.html", self_classes=self_classes)


@app.route("/augmentations/preview", methods=["POST"])
def previewAugmentations():
    req = request.get_json()
    image = cv2.imread("src"+req['image'])
    augmented = get_preview([image], req['augmentationList'], 1)
    cv2.imwrite('src/static/images/temps/preview_aug.png', augmented[0])
    return make_response(jsonify({'message': 'Preview saved to static/images/temps/preview_aug.png'}), 200)


@app.route("/augmentations/uploadimage", methods=["POST"])
def uploadAugmentationImage():
    if request.method == 'POST':
        image = request.files['file']
        path = "/static/images/temps/upload_aug.png"
        image.save("src/"+path)
        return make_response(jsonify({'message': 'Uploaded image successfully', 'path': path}), 200)


@app.route("/augmentations/savechanges", methods=["POST"])
def applyAugmentations():
    if request.method == 'POST':
        req = request.get_json()
        percentage = req['percentage']/100
        auglist = req['augmentationList']
        classes = req['classList']
        path_gtsrb = app.config["DATA_PATH"] + "dataset/GTSRB/train/"
        path_self = app.config["DATA_PATH"] + "dataset/New/Train/"
        for i in classes:
            saveAugmentedImages(i, path_gtsrb + str(i).rjust(4, '0'), auglist, percentage)
            saveAugmentedImages(i, path_self + str(i), auglist, percentage)
        return redirect("/augmentations")


@app.route("/dataset", methods=["GET", "POST"])
def datasetStatistics():
    base_path = app.config["DATA_PATH"] + "dataset/"

    # GTSRB
    base_path_GTSRB = os.path.join(base_path, "GTSRB/")
    # train_set (GTSRB)
    train_path_GTSRB = os.path.join(base_path_GTSRB, "train/")
    train_class_dist_gtsrb = []
    for i in range(len(orig_classes)):
        folder = "{:04d}".format(i)
        path = train_path_GTSRB + folder + "/GT-" + folder + ".csv"
        if os.path.exists(path):
            with open(path, 'r') as csv_file:
                csvreader = csv.reader(csv_file, delimiter=';')
                row_count = sum(1 for row in csvreader)
                train_class_dist_gtsrb.insert(i, (row_count-1))
        else:
            train_class_dist_gtsrb.insert(i, 0)
    #val_set (GTSRB)
    val_path_GTSRB = os.path.join(base_path_GTSRB, "valid/")
    val_class_dist_gtsrb = []
    for i in range(len(orig_classes)) :
        folder = "{:04d}".format(i) 
        path = val_path_GTSRB + folder + "/GT-" + folder + ".csv"
        if os.path.exists(path) :
            with open(path, 'r') as csv_file:
                csvreader = csv.reader(csv_file, delimiter=';')
                row_count = sum(1 for row in csvreader)
                val_class_dist_gtsrb.insert(i,(row_count-1))
        else :
            val_class_dist_gtsrb.insert(i, 0)
    #test_set (GTSRB)
    test_path_GTSRB = os.path.join(base_path_GTSRB, "test/")
    test_class_dist_gtsrb = [0] * len(orig_classes) 
    path = test_path_GTSRB + "GT-Test.csv"
    with open(path, 'r') as csv_file:
        csvreader = csv.reader(csv_file, delimiter=';')
        next(csvreader)
        for row in csvreader: 
            test_class_dist_gtsrb[int(row[1])] += 1

    #Extra
    base_path_Extra = os.path.join(base_path, "EXTRA/")
    # train_set (EXTRA)
    train_path_Extra = os.path.join(base_path_Extra, "train/")
    train_class_dist_extra = []
    for i in range(0, len(orig_classes) + len(self_classes)):
        folder = "{:04d}".format(i)
        path = train_path_Extra + folder + "/GT-" + folder + ".csv"
        if os.path.exists(path):
            with open(path, 'r') as csv_file:
                csvreader = csv.reader(csv_file, delimiter=';')
                row_count = sum(1 for row in csvreader)
                train_class_dist_extra.insert(i, (row_count-1))
        else:
            train_class_dist_extra.insert(i, 0)

    val_class_dist_extra = []
    test_class_dist_extra = []

    return render_template("dataset.html", orig_classes_count=len(orig_classes), self_classes_count=len(self_classes), train_class_dist_gtsrb = train_class_dist_gtsrb, train_class_dist_extra = train_class_dist_extra, count_org = sum(train_class_dist_gtsrb), count_new = sum(train_class_dist_extra), val_class_dist_gtsrb = val_class_dist_gtsrb, val_class_dist_extra = val_class_dist_extra, test_class_dist_gtsrb = test_class_dist_gtsrb, test_class_dist_extra = test_class_dist_extra,)


@app.route("/training", methods=["GET", "POST"])
def trainModel():
    pretrained_models = []
    for file in os.listdir(app.config["DATA_PATH"] + "checkpoints/logs/micronet_params"):
        print(file)
        if file.endswith(".pt"):
            pretrained_models.append(file)
    print(pretrained_models)
    return render_template("training.html", self_classes=self_classes, pretrained_models=pretrained_models)


@app.route("/training/train", methods=["POST"])
def ModelTraindata():
    data = request.get_json()
    with open(app.config["DATA_PATH"] + "/config/params.json") as json_file:
        default_configs = json.load(json_file)
    default_configs["experiment"]["model"] = data["model"]
    default_configs["experiment"]["batch_size"] = int(data["batch_size"])
    default_configs["experiment"]["epochs"] = int(data["epochs"])
    default_configs["experiment"]["learning_rate"] = float(
        data["learning_rate"])
    default_configs["experiment"]["class_ids"] = list(
        range(0, 43)) + (np.array(data["class_ids"]) + 43).tolist()
    default_configs["experiment"]["num_classes"] = int(
        data["num_classes"]) + 43
    with open(app.config["DATA_PATH"] + "config/temp_config.json", "w") as outfile:
        json.dump(default_configs, outfile)

    shutil.move(os.path.join(app.config["DATA_PATH"], "dataset/EXTRA"),
              os.path.join(app.config["DATA_PATH"], "dataset/EXTRA_copy"))
    remake_EXTRA_folder(
        0, 0, new_classes=default_configs["experiment"]["class_ids"])
    remake_EXTRA_folder(
        0, 1, new_classes=default_configs["experiment"]["class_ids"])
    remake_EXTRA_folder(
        1, 0, new_classes=default_configs["experiment"]["class_ids"])
    train.train("temp_config")
    shutil.rmtree(app.config["DATA_PATH"] + "dataset/EXTRA")
    shutil.move(os.path.join(app.config["DATA_PATH"], "dataset/EXTRA_copy"),
              os.path.join(app.config["DATA_PATH"], "dataset/EXTRA"))

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


@app.route("/validation", methods=["GET", "POST"])
def createValidationSet():
    return render_template("validation.html", self_classes=self_classes)


@app.route("/validation/uploadimages", methods=["GET", "POST"])
def uploadValidationImages():
    if request.method == 'POST':
        cnt = int(request.form['cnt'])
        classId = request.form['class']
        images = []
        for i in range(cnt):
            images.append(request.files[f'file{i}'])
        if upload_images("Valid", images, "", classId):
            print("uploaded successfully")
        else:
            print("failed to upload")
        # images = request.files['files']
        # print(images)
        # path = "/static/images/temps/upload_aug.png"
        # image.save("src/"+path)
        return make_response(jsonify({'message': 'Uploaded images successfully'}), 200)


@app.route("/validation/segregation", methods=["POST"])
def smartSegregation():
    if request.method == 'POST':
        req = request.get_json()
        splitRatio = req['splitRatio']
        remake_EXTRA_folder(splitRatio, 0)
        return make_response(jsonify({'message': 'Uploaded images successfully'}), 200)


@app.route("/results", methods=["GET", "POST"])
def viewResults():
    with open(app.config["JSON_PATH"] + "metrics.json") as json_file:
        data = json.load(json_file)
    conf_arr = numpy.load(app.config["JSON_PATH"] + "confusion_matrix.npy")
    num_class = numpy.shape(conf_arr)[0]

    return render_template("results.html", data=data, matrix=conf_arr, class_count=num_class)

# helpers


def upload_images(dataset, file_list, class_name, classID=None):
    try:
        class_id = classID
        if class_id is None:
            class_id = get_index(orig_classes, class_name)
            if class_id != -1:
                class_id = str(class_id)
            else:
                class_id = str(get_index(self_classes, class_name) + 43)

        path = app.config["DATA_PATH"] + \
            "dataset/New/" + dataset + "/" + class_id + "/"
        if os.path.exists(path) == False:
            os.mkdir(path)
        for f in file_list:
            f.save(os.path.join(path, f.filename))
        if dataset == "Train":
            remake_EXTRA_folder(0, 0)
        elif dataset == "Test":
            remake_EXTRA_folder(0, 1)
        else:
            remake_EXTRA_folder(1, 0)
        return True
    except Exception as e:
        print(e)
        return False


def delete_classes(class_list, folders=["Train", "Test", "Valid"]):
    for klass in class_list:
        class_id = get_index(self_classes, klass) + 43
        for folder in folders:
            dataset_dir = app.config["DATA_PATH"] + \
                "dataset/New/" + folder + "/"
            path = os.path.join(dataset_dir, str(class_id))
            if(os.path.exists(path)):
                shutil.rmtree(path)
            class_dirs = os.listdir(dataset_dir)
            for dir in class_dirs:
                if dir == ".gitkeep":
                    continue
                if int(dir) > class_id:
                    os.rename(os.path.join(dataset_dir, dir), os.path.join(
                        dataset_dir, str(int(dir)-1)))
        self_classes.remove(klass)
    save_self_classes_json()
    # make the EXTRA folder again
    remake_EXTRA_folder(0, 0)
    remake_EXTRA_folder(0, 1)
    remake_EXTRA_folder(1, 0)


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


def remake_EXTRA_folder(val_fraction, test_fraction, new_classes=[]):
    extra_path = os.path.join(app.config["DATA_PATH"], "dataset/EXTRA/")
    new_path = os.path.join(app.config["DATA_PATH"], "dataset/New/")
    if val_fraction == 1:
        new_path += "Valid/"
    elif test_fraction == 1:
        new_path += "Test/"
    else:
        new_path += "Train/"
    # shutil.rmtree(extra_path)
    # os.mkdir(extra_path)
    prepare_new_classes.prepare_train_val_n_test(
        new_path, extra_path, validation_fraction=val_fraction, test_fraction=test_fraction, new_classes=new_classes)
    return


def saveAugmentedImages(classID, path, auglist, percentage):
    path_augmented = app.config["DATA_PATH"] + "dataset/New/Augmented/"
    image_names = []
    image_list = []
    if os.path.exists(path):
        images = os.listdir(path)
        for image in images:
            name, ext = image.split('.')
            if ext not in ['csv', 'gitignore', 'gitkeep']:
                image_list.append(cv2.imread(f'{path}/{image}'))
                image_names.append(name)
        augmented = get_preview(
            image_list, auglist, percentage)

        save_path = path_augmented + str(classID)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for name, image in zip(image_names, augmented):
            cv2.imwrite(f'{save_path}/{name}_augmented.ppm', image)

        image_list.clear()
        image_names.clear()
