# DETEKTOR

[Link](https://www.thedetektor.com/) to the project website.

[Link](data/traffic_sign_interiit/README.md) to README of DL code.

## INSTRUCTIONS TO RUN THE UI

Python version: 3.7.10

Open a terminal in the root directory of the project
```
python -m pip install -r requirements.txt
python app.py
```
The UI can be accessed at `localhost:5000` on any commonly used browser.


## FEATURES

- Add Images
- Add Augmentations
- Create Validation Set
- View Dataset Statistics
- Visualise Model
- Train and Test Model
- View Results

## FEATURE DESCRIPTIONS

### 1. Add Images
Using the *Add/Delete Custom Classes* section in this tab, new custom classes can be added to our class set. Then using the *Upload Images* section, we can add images for these newly added custom classes.

We can also add new images to existing classes of the GTSRB dataset.

Preview of images of the GTSRB classes and customly added classes can also be viewed on this page.

Usage: 
- Upload Images:
    1. Choose the set to which data is to be added. (Train or Test)
    2. Select the Class for which the data is to be added. (Includes Original + New classes)
    3. Select image(s) to be uploaded and click Upload.
- Add/Delete Custom Classes:
    1. Type name of new class and click **Add Class** button
    2. To delete new classes select them and click **Delete Class** button

        **NOTE**
        Along with the new class, all data added for it will also be deleted

### 2. Add Augmentations
This page provides the functionality to:
1. Test and Preview augmentations on sample images.
2. Apply the selected augmentations to the dataset.

Usage: 
1. Select an image to test augmentations on either from the existing samples or by uploading an image of your choice using the **Upload Image** button. 
2. To add any augmentation, first of all select the class whose augmentation needs to be applied followed by selecting the appropriate augmentation. Set the relevant parameters and click on the **Add augmentation** button. It will now be visible under the Applied Augmentations column. 
3. Clicking on the **Preview** button will apply the augmentations to the selected image and the new image will be shown. 
4. Once the augmentations are finalised we can apply them to the dataset. Click on the **Apply Augmentations** button, a popup window will appear. The user can select the classes to which the augmentations should be applied to along with the probability which determines the percentage of total images on which the augmentation will be applied. 
5. Clicking on Save changes will apply the augmentations to the dataset.

### 3. Create Validation Set
This page provides the functionality to:
1. Add validation images for any class
2. Split training into training and validation based on a specific split ratio

### 4. View Dataset Statistics :
In this tab we can see all the dataset statistics of our GTSRB dataset and the new data added using *Add Images* tab and augmented using *Add Augmentations* tab.

Structure:
- On top there are two cards showing total labels and total data respectively.
- Then, there is a Pie chart showing the data distribution among Train, Validation and Test set.
- Then there are three stacked bar charts showing the individual distribution of each set into GTSRB data, augmented data and data of newly added classes.

### 5. Visualise Model  :
This tab is dedicated to get a visual view of our model. An open source program called “Netron” is used to achieve this.

Usage: 
1. Select a model from the dropdown list and click the **Visualise Model** button.

### 6. Train and Test Model :
This page has two functionalities:
1. Training a new model: Users can train a new model on specific additional classes and change the hyperparameters.
2. Testing a pre-trained model: Users can test their pre-trained models on the classes they trained it on and  see the accuracy results.

Usage:
- Training a new model:
    1. In the *Train Model* section, select the additional classes you want to train the model on from Select Classes textboxes. If none are selected, the model will train on the pure GTSRB dataset.
    2. Select the hyperparameters: Batch Size, Epochs from the drop down and enter the Learning Rate.
    3. Click on the **Train Model** button to start training.
- Testing a pre-trained model:
    1. In the *Test Model* section, select a pre-trained model from the drop-down list and click on the **Test Model** button. 

### 7. View Results :
We can view obtained results in this tab. 
- Structure:
    1. Option to select the model the user wants to see results for.
    2. Then, there is a table having the classification accuracy and other metrics of that model for each class.
    3. A **See confusion matrix** button will show a popup showing the confusion matrix which will be a better visual for the classification accomplishment.
    4. Rise maps of some correctly and incorrectly classified traffic signs.
