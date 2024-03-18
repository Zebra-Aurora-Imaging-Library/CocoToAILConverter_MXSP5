# Import dependencies.
import mil as MIL
from pycocotools.coco import COCO
import numpy as np

# -------------------- VARIABLES OF THE CODE  ------------------------------------------------
LABEL_FILE = 'COCOdataset/coco-wood.json'
DESTINATION_FOLDER_LABEL_IMAGES = 'MilLabels'
# If the FILTER_CLASSES is empty, this code will generate labels for all the images using all the classes.
# Class of interest can be selected by adding them to the list.
FILTER_CLASSES = []
# --------------------------------------------------------------------------------------------

# Initialize the COCO API for instance annotations.
coco = COCO(LABEL_FILE)
category_IDs = coco.getCatIds()
categories = coco.loadCats(category_IDs)

# Create a category name and category ID dictionary for later use.
category_name_and_id = {}
for i in categories:
    category_name_and_id[i['id']] = i['name']

# If FILTER_CLASSES is empty, make it a list of all class names.
if len(FILTER_CLASSES) == 0:
    ALL_CLASSES = []
    for i in categories:
        ALL_CLASSES = ALL_CLASSES + [i['name']]
    FILTER_CLASSES = ALL_CLASSES


# Fetch class IDs that have one or more classes in the FILTER_CLASSES.
img_IDs = []
# Iterate for each individual class in the list.
for class_name in FILTER_CLASSES:
    # get all images containing given class
    cat_IDs = coco.getCatIds(catNms=class_name)
    img_IDs = img_IDs + coco.getImgIds(catIds=cat_IDs)

# Remove duplicates.
img_IDs_filtered = list(set(img_IDs))
category_IDs_filtered = coco.getCatIds(catNms=FILTER_CLASSES)

# Allocate MIL resources.
MilApplication = MIL.MappAlloc("M_DEFAULT", MIL.M_DEFAULT)
MilSystem = MIL.MsysAlloc(MilApplication, "M_DEFAULT", MIL.M_DEFAULT, MIL.M_DEFAULT)

# Populate dummy dataset to generate LUT for labels.
num_classes = 256
MilDummyDataset = MIL.MclassAlloc(MilSystem, MIL.M_DATASET_IMAGES, MIL.M_DEFAULT, MIL.M_DEFAULT)
for i in range (num_classes):
    MIL.MclassControl(MilDummyDataset, MIL.M_DEFAULT, MIL.M_CLASS_ADD, MIL.M_NULL)

# LUT for labeled image.
MilLut = MIL.MbufAllocColor(MilSystem, 3, num_classes, 1, 8 + MIL.M_UNSIGNED, MIL.M_LUT, MIL.M_DEFAULT)
MIL.MclassDraw(MIL.M_DEFAULT, MilDummyDataset, MilLut, MIL.M_DRAW_CLASS_COLOR_LUT, MIL.M_DEFAULT, MIL.M_DEFAULT)

for i in range(len(img_IDs_filtered)):
    # Get the current image.
    current_img = coco.loadImgs(img_IDs_filtered[i])[0]

    # Get the annotation of the current image.
    current_annotation_IDs = coco.getAnnIds(imgIds=current_img['id'], catIds=category_IDs_filtered, iscrowd=None)
    current_annotations = coco.loadAnns(current_annotation_IDs)

    # Change the pixel value according to the index of the category in the filterClass.
    label_image = np.zeros((current_img['height'], current_img['width']))
    for j in range(len(current_annotations)):
        class_name = category_name_and_id[current_annotations[j]['category_id']]
        pixel_value = FILTER_CLASSES.index(class_name) + 1
        label_image = np.maximum(coco.annToMask(current_annotations[j]) * pixel_value, label_image)

    # Change the format to make it compatible with the MIL buffer.
    label_image = label_image.astype(np.uint8)
    size_x = current_img['width']
    size_y = current_img['height']
    MilLabelImg = MIL.MbufAllocColor(MilSystem, 1, size_x, size_y, 8 + MIL.M_UNSIGNED, MIL.M_IMAGE + MIL.M_PROC + MIL.M_DISP, MIL.M_DEFAULT)
    MIL.MbufPut(MilLabelImg, label_image)

    # Associate the lookup table to the image.
    MIL.MbufControl(MilLabelImg, MIL.M_ASSOCIATED_LUT, MilLut)

    # Save the label image.
    MIL.MbufSave(DESTINATION_FOLDER_LABEL_IMAGES + '/' + current_img['file_name'], MilLabelImg)
    MIL.MbufFree(MilLabelImg)

# Free the MIL resources.
MIL.MclassFree(MilDummyDataset)
MIL.MbufFree(MilLut)
MIL.MsysFree(MilSystem)
MIL.MappFree(MilApplication)