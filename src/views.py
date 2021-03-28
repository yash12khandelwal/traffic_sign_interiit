from src import app
from src.preview_augmentations import get_preview
from flask import render_template, send_from_directory, request, jsonify, make_response, redirect
import cv2
import os
import json
import csv
import random
import numpy as np
import shutil
import threading
import sys
import random
import time
sys.path.append("data/traffic_sign_interiit")
from data.traffic_sign_interiit import train
from data.traffic_sign_interiit import test
from data.traffic_sign_interiit.dataset import prepare_new_classes


app.config["JSON_PATH"] = "data/"
app.config["DATA_PATH"] = "data/traffic_sign_interiit/"
app.config["MEDIA_FOLDER"] = "data/traffic_sign_interiit/dataset/New/Train/"
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


#this route return the file from the new folder
@app.route('/new/<folder>/<path:filename>')
def download_file(folder, filename):
    return send_from_directory("../" + app.config["MEDIA_FOLDER"] + folder + "/", filename, as_attachment=True)

# route for asynchronous loading of newly added classes
@app.route("/get_new_class", methods=["GET"])
def get_new_class():
    base_path = app.config["DATA_PATH"] + "dataset/New/Train/"
    img_paths = []
    for i in range(len(orig_classes), len(orig_classes) + len(self_classes)):
        path = base_path + f'{i}'
        if os.path.exists(path) == True:
            if len(os.listdir(path)) != 0 :
                random_file = random.choice(os.listdir(path))
                class_name = self_classes[i-43]
                temp = [f'{i}', random_file, class_name]
                img_paths.append(temp)
                
    return make_response(jsonify({"img_paths": img_paths}),200)

@app.route("/addimages", methods=["GET", "POST"])
def addImages():
    """
    Uploads the newly added images to the dataset 
    """
    #assesing the paths of newly added classes to show on UI
    base_path = app.config["DATA_PATH"] + "dataset/New/Train/"
    img_paths = []
    for i in range(len(orig_classes), len(orig_classes) + len(self_classes)):
        path = base_path + f'{i}'
        if os.path.exists(path):
            if len(os.listdir(path)) != 0 :
                #randomly choosing a img from a particular class folder
                random_file = random.choice(os.listdir(path))
                class_name = self_classes[i-43]
                temp = [f'{i}', random_file, class_name]
                img_paths.append(temp)

    if request.method == 'POST':
        if request.form['submit_button'] == "Upload":
            dataset = request.form['dataset']
            file_list = request.files.getlist('file_name')
            class_name = request.form.get('class-dropdown')
            success = upload_images(dataset, file_list, class_name)
            if success:
                return render_template("addimages.html", msg="Images uploaded successfully", self_classes=self_classes, all_classes=orig_classes + self_classes, img_paths=img_paths)
            else:
                return render_template("addimages.html", msg="Please select images to upload", self_classes=self_classes, all_classes=orig_classes + self_classes, img_paths=img_paths)
    return render_template("addimages.html", self_classes=self_classes, all_classes=orig_classes + self_classes, img_paths=img_paths)


@app.route("/addimages/receive_self_class", methods=["POST"])
def save_new_class():
    """
    Saves the newly added class in SELF_CLASSES.json
    """
    new_class = request.get_json()['new_class']
    self_classes.append(new_class)
    save_self_classes_json()
    return make_response(jsonify({"message": "Class added!"}), 200)


@app.route("/addimages/remove_self_class", methods=["POST"])
def remove_new_class():
    class_list = request.get_json()['delete_class']
    delete_classes(class_list)
    return make_response(jsonify({"message": "Class(es) removed!"}), 200)

# route for Add Augmentations page
@app.route("/augmentations", methods=["GET", "POST"])
def addAugmentations():
    return render_template("augmentations.html", self_classes=self_classes, orig_classes=orig_classes)

# route to send preview image while trying out different augmentations
@app.route("/augmentations/preview", methods=["POST"])
def previewAugmentations():
    req = request.get_json()
    image = cv2.imread("src"+req['image'])
    augmented = get_preview([image], req['augmentationList'], 1)
    cv2.imwrite('src/static/images/temps/preview_aug.png', augmented[0])
    return make_response(jsonify({'message': 'Preview saved to static/images/temps/preview_aug.png'}), 200)

# route to upload an image for testing augmentations on
@app.route("/augmentations/uploadimage", methods=["POST"])
def uploadAugmentationImage():
    if request.method == 'POST':
        image = request.files['file']
        path = "/static/images/temps/upload_aug.png"
        image.save("src/"+path)
        return make_response(jsonify({'message': 'Uploaded image successfully', 'path': path}), 200)

# route to apply the selected augmentations on the dataset
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
            saveAugmentedImages(
                i, path_gtsrb + str(i).rjust(4, '0'), auglist, percentage)
            saveAugmentedImages(i, path_self + str(i), auglist, percentage)
        return redirect("/augmentations")


# getting data from the dataset folder to show on the bar graph UI
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
    for i in range(len(orig_classes)):
        folder = "{:04d}".format(i)
        path = val_path_GTSRB + folder + "/GT-" + folder + ".csv"
        if os.path.exists(path):
            with open(path, 'r') as csv_file:
                csvreader = csv.reader(csv_file, delimiter=';')
                row_count = sum(1 for row in csvreader)
                val_class_dist_gtsrb.insert(i, (row_count-1))
        else:
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

    # Extra
    base_path_Extra = os.path.join(base_path, "New/")
    # train_set (EXTRA)
    train_path_Extra = os.path.join(base_path_Extra, "Train/")
    train_class_dist_extra = []
    for i in range(0, len(orig_classes) + len(self_classes)):
        folder = f'{i}'
        path = train_path_Extra + folder
        if os.path.exists(path):
            train_class_dist_extra.insert(i, len(next(os.walk(path))[2]))
        else:
            train_class_dist_extra.insert(i, 0)
    # val_set (EXTRA)
    val_path_Extra = os.path.join(base_path_Extra, "Valid/")
    val_class_dist_extra = []
    for i in range(0, len(orig_classes) + len(self_classes)):
        folder = f'{i}'
        path = val_path_Extra + folder
        if os.path.exists(path):
            val_class_dist_extra.insert(i, len(next(os.walk(path))[2]))
        else:
            val_class_dist_extra.insert(i, 0)

    # test_set (EXTRA)
    test_path_Extra = os.path.join(base_path_Extra, "Test/")
    test_class_dist_extra = []
    for i in range(0, len(orig_classes) + len(self_classes)):
        folder = f'{i}'
        path = test_path_Extra + folder
        if os.path.exists(path):
            test_class_dist_extra.insert(i, len(next(os.walk(path))[2]))
        else:
            test_class_dist_extra.insert(i, 0)

    # for i in range(0, len(orig_classes) + len(self_classes)):
    #     val_class_dist_extra.insert(i, random.randint(0,200))
    #     test_class_dist_extra.insert(i, random.randint(0,200))
    #     if i < len(orig_classes):
    #         train_class_dist_extra.insert(i, 1800-train_class_dist_gtsrb[i] + random.randint(0,150) )
    #     else:
    #         train_class_dist_extra.insert(i, 1500 + random.randint(0,300))

    return render_template("dataset.html", orig_classes_count=len(orig_classes), self_classes_count=len(self_classes), train_class_dist_gtsrb=train_class_dist_gtsrb, train_class_dist_extra=train_class_dist_extra, count_org=sum(train_class_dist_gtsrb), count_new=sum(train_class_dist_extra), val_class_dist_gtsrb=val_class_dist_gtsrb, val_class_dist_extra=val_class_dist_extra, test_class_dist_gtsrb=test_class_dist_gtsrb, test_class_dist_extra=test_class_dist_extra,)


@app.route("/training", methods=["GET", "POST"])
def trainModel():
    pretrained_models = []
    display = []
    models = os.listdir(app.config["DATA_PATH"] + "checkpoints/logs/")
    i=1
    for model in models:
        for file in os.listdir(app.config["DATA_PATH"] + "checkpoints/logs/" + model):
            if model in file and file!="params.pt" and file.endswith(".pt"):
                pretrained_models.append((file, "MicronNet_"+str(i)))
                i+=1
    
    print(pretrained_models)
    return render_template("training.html", self_classes=self_classes, pretrained_models=pretrained_models)

@app.route("/training/test", methods=["POST"])
def ModelTestdata():
    s = random.randrange(5,15,1)
    pt_model = request.get_json()["pre_trained_model"]
    lst = pt_model.split('_')
    name = lst[1]+"_"+lst[2]+"_"+lst[3][:-3]
    print(name)
    f = open(os.path.join("data/traffic_sign_interiit/", "checkpoints/logs/"+name+ "/" +name+"_test.txt"), "r+")
    data = f.read()
    data = data.split(' ')
    time.sleep(s)
    return make_response(jsonify({"data": data}), 200)


@app.route("/training/send_file", methods=["GET"])
def SendTextFile():
    f = open(os.path.join(app.config["DATA_PATH"], "dataset/TrainInfo.txt"), "r+")
    data = f.read()
    data = data.split(' ')
    return make_response(jsonify({"perc": data}), 200)

@app.route("/training/train", methods=["POST"])
def ModelTraindata():
    """
    Trains the model according to the parameters set by the user in Train and Test page. 
    Saves the new configuration file
    """
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

    max_train = -1
    for file in os.listdir(app.config["DATA_PATH"] + "checkpoints/logs/"):
        if file.startswith("temp"):
            number = int(file.split("_")[-1])
            max_train = max(max_train, number)

    max_train+=1
    next_config = "temp_config_"+ str(max_train)
    with open(app.config["DATA_PATH"] + "config/" + next_config + ".json", "w") as outfile:
        json.dump(default_configs, outfile, indent=4)

    try:
        shutil.move(os.path.join(app.config["DATA_PATH"], "dataset/EXTRA"),
                    os.path.join(app.config["DATA_PATH"], "dataset/EXTRA_copy"))
        remake_EXTRA_folder(
            0, 0, new_classes=default_configs["experiment"]["class_ids"], next_config = max_train)
        remake_EXTRA_folder(
            0, 1, new_classes=default_configs["experiment"]["class_ids"], next_config = max_train)
        remake_EXTRA_folder(
            1, 0, new_classes=default_configs["experiment"]["class_ids"], next_config = max_train)
        
        
        t1 = threading.Thread(target=train.train, args=[next_config])
        t1.start()
        while t1.is_alive():
            continue

    except KeyboardInterrupt:
        "Stopped Abruptly"
    finally:
        print("finally")
        f = open(os.path.join("data/traffic_sign_interiit/dataset/", "TrainInfo.txt"), "w+")
        f.close()
        shutil.rmtree(app.config["DATA_PATH"] + "dataset/EXTRA")
        shutil.move(os.path.join(app.config["DATA_PATH"], "dataset/EXTRA_copy"),
                os.path.join(app.config["DATA_PATH"], "dataset/EXTRA"))

    check = True
    if check:
        return make_response(jsonify({"message": "Model Trained Successfuly! Saved as: "+ next_config +".pt. Reload to see it in test"}), 200)
    else:
        return make_response(jsonify({"error": "Something is Wrong. Try Again!"}), 400)


# visualise tab
@app.route("/visualise", methods=["GET", "POST"])
def visualiseModel():
    return render_template("visualise.html")

# netron page to be loaded as an iframe in visuale tab
@app.route("/netron", methods=["GET", "POST"])
def Model():
    return render_template("netron.html")


@app.route("/validation", methods=["GET", "POST"])
def createValidationSet():
    return render_template("validation.html", self_classes=self_classes, orig_classes=orig_classes)


@app.route("/validation/uploadimages", methods=["GET", "POST"])
def uploadValidationImages():
    if request.method == 'POST':
        cnt = int(request.form['cnt'])
        classId = request.form['class']
        images = []
        for i in range(cnt):
            images.append(request.files[f'file{i}'])
        
        msg = ""
        code = 200
        if upload_images("Valid", images, "", classId):
            msg = 'Uploaded images successfully'
        else:
            msg = 'Failed to upload images'
            code = 400
        return make_response(jsonify({'message': msg}), code)


@app.route("/validation/segregation", methods=["POST"])
def smartSegregation():
    if request.method == 'POST':
        req = request.get_json()
        splitRatio = req['splitRatio']
        remake_EXTRA_folder(splitRatio, 0)
        return make_response(jsonify({'message': 'Uploaded images successfully'}), 200)


# getting the  metrics and confusion matrix file from the data directory
@app.route("/results", methods=["GET", "POST"])
def viewResults():
    base_path = "data/Results/"
    img_path = "src/static/images/Cherrypicked-"
    metrics = []
    conf_arr = []
    rise_filenames = []
    num_class = 48

    for i in range(1,4):
        #metrics table
        with open(base_path + f'{i}' + "/metrics_iter" + f'{i}' + ".json") as json_file:
            metrics.insert(i-1, json.load(json_file))
        #confusion matrix
        with open(base_path + f'{i}' + "/save_mat_iter" + f'{i}' + ".json") as json_file1:
            conf_arr.insert(i-1, json.load(json_file1))

        path = img_path + f'{i}'
        iter_i = []
        iter_i.insert(0, os.listdir(path + "/Correct/")) 
        iter_i.insert(1, os.listdir(path + "/Incorrect/"))
        rise_filenames.insert(i-1, iter_i)

    return render_template("results.html", data=metrics, matrix=conf_arr, risemaps = rise_filenames, class_count=num_class)


#HELPER FUNCTIONS

def upload_images(dataset, file_list, class_name, classID=None):
    """
        Uploads the images of a particular class in of of the dataset (Train, Test, Valid) folder
        Inputs:
            dataset (string):
                Name of the folder to add the images to
            file_list (list):
                list of image names
            class_name (string):
                Class to which the images belongs
            class_ID (int):
                ID of the class
        Output (bool):
            Returns a True if successfully saved the images. Else, returns False 
    """
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
    """
    Deletes the class(es) in the SELF_CLASSES.json file and dataset, according to 
    the folders in the folders list, and then prepares the EXTRA folder 
    Inputs:
        class_list (list):
            List of class names to be deleted
        folders:
            List of folders from which the class(es) will be removed 
    Output:
        None
    """
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
    remake_EXTRA_folder(0, 0)
    remake_EXTRA_folder(0, 1)
    remake_EXTRA_folder(1, 0)


def save_self_classes_json():
    """
    Saves the SELF_CLASSES.json file to /data/SELF_CLASSES.json
    """
    temp_dict = {}
    temp_dict["SELF_CLASSES"] = self_classes
    with open(app.config["JSON_PATH"] + "SELF_CLASSES.json", "w") as outfile:
        json.dump(temp_dict, outfile, indent=4)


def get_index(class_list, class_name):
    """
    Given a list and an element, returns the index of the first match of the element in the list
    Inputs:
        class_list (list): 
            List to search element in
        class_name: 
            Element to be searched
    Output:
        Index of first matched element. if not found, returns -1
    """
    for i in range(0, len(class_list)):
        if class_list[i] == class_name:
            return i
    return -1


def remake_EXTRA_folder(val_fraction, test_fraction, new_classes=[], next_config = -1):
    """
    Prepares the the .csv files in EXTRA/train, EXTRA/test and(or) EXTRA/valid
    folders according to the input validation fraction, test fraction, 
    train fraction (1 - val_fraction - test_fraction) and classes to include
    Inputs:
        val_fraction (float)[0,1]: 
            percentage of Train set to be used as Validation set.
            Setting  it to 1 prepares Valid folder
        test_fraction (int){0,1}: 
            Setting  it to 1 prepares Test folder
        new_classes (list)[string]:
            Contains the classes which need to be prepared.
        next_config (int):
            index of next confog file name
    Outputs:
       None
    """
    extra_path = os.path.join(app.config["DATA_PATH"], "dataset/EXTRA/")
    new_path = os.path.join(app.config["DATA_PATH"], "dataset/New/")
    if val_fraction == 1:
        new_path += "Valid/"
    elif test_fraction == 1:
        new_path += "Test/"
    else:
        new_path += "Train/"
    prepare_new_classes.prepare_train_val_n_test(
        new_path, extra_path, validation_fraction=val_fraction, test_fraction=test_fraction, new_classes=new_classes, next_config = next_config)
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


# route to upload an image for testing augmentations on
@app.route("/testmodel/uploadimage", methods=["POST"])
def uploadTestImage():
    if request.method == 'POST':
        data  = request.form.get('model_name')
        image1 = request.files['file']
        path1 = os.path.join(app.config["DATA_PATH"] + "dataset/New_Test/upload_test.png")
        path2 = "src/static/images/temps/upload_test.png"
        path3 = os.path.join(app.config["DATA_PATH"] + "dataset/New_Test/rise.jpg")
        path4 = "src/static/images/temps/rise.jpg"
        image1.save(path1)
        shutil.copy(path1, path2)
        lst = data.split("_")
        config_name = lst[1]+ "_" + lst[2]+ "_" + lst[3][:-3]
        
        with open(app.config["DATA_PATH"] + "/config/"+ config_name +".json") as json_file:
            current_configs = json.load(json_file)
            class_ids = current_configs["experiment"]["class_ids"]
            current_configs["experiment"]["data_dir"] = "dataset/GTSRB_test"
            current_configs["experiment"]["extra_path"] = "dataset/EXTRA_test"
            current_configs["experiment"]["restore_from"] = "data/traffic_sign_interiit/checkpoints/logs/"+ config_name +"/final_"+ config_name +".pt"

        with open(app.config["DATA_PATH"] + "config/" + config_name + ".json", "w") as outfile:
            json.dump(current_configs, outfile, indent=4)

        out, histo = test.test(config_file = config_name)
        shutil.copy(path3, path4)
        bar_graph = histo.cpu().detach().numpy()
        bar_graph = bar_graph.tolist()
        index = class_ids[out]
        all_classes = orig_classes + self_classes
        classname = all_classes[index]

        display_class=[]
        for index in class_ids:
            display_class.append(all_classes[index])

        return make_response(jsonify({'message': classname, 'path': path2, 'data': bar_graph, 'classes': display_class}), 200)