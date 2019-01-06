import cv2
import os
import numpy as np

# this array will hold the persons' names to identify who the person is after recognition
subjects = ["", "Duda", "Merkel", "Obama", "Trump", "Putin"]


# ### Prepare training data ###

# detect face using OpenCV
def detect_face(img):
    # convert the test image to grayscale as opencv face detector works with gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # load OpenCV face detector, with open cv haar classifier
    face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    # no face detected
    if (len(faces) == 0):
        return None, None

    # extract the face area as region of interest, considering there is only one face in image
    (x, y, w, h) = faces[0]

    # return the face part of the image
    return gray[y: y + w, x: x + h], faces[0]


# preparing training data. This includes cutting faces out of images, storing and labeling them.
def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)

    faces = []
    labels = []

    # for each img in each directory from dataset:
    for dir_name in dirs:

        # if directory name does not start with "person" its irrelevant data.
        if not dir_name.startswith("person"):
            continue;

        # Remove "person" part from the directory name, the result is id(or label) of the person. Consider our persons' name array. These ids will be index of those array.
        label = int(dir_name.replace("person", ""))
        subject_dir_path = data_folder_path + "/" + dir_name
        # Matching label with related name
        subject_images_names = os.listdir(subject_dir_path)

        # for each image name in each folder read it, extract face and add to list
        for image_name in subject_images_names:

            # ignore system files
            if image_name.startswith("."):
                continue;

            image_path = subject_dir_path + "/" + image_name

            # read image
            image = cv2.imread(image_path)

            # show image
            cv2.imshow("Training...", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)

            # detect face
            face, rect = detect_face(image)

            if face is not None:
                # add face to list of faces
                faces.append(face)
                # append label to face
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels


print("Preparing data...")
faces, labels = prepare_training_data("images")
print("Data prepared")

# print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

# ### Train Face Recognizer ###

# In this project we will use cv2's lbph face recognizer.
# LBPH algorithm creates histograms for each image. Using these histograms and the sample image's histogram we will calculate distance between two of them and try to recognize image.
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Alternatives:

# EigenFaces Method for recognition
# face_recognizer = cv2.face.EigenFaceRecognizer_create()

# Fisher Method for recognition
# face_recognizer = cv2.face.FisherFaceRecognizer_create()


# Now that we have initialized our face recognizer and we also have prepared our training data, it's time to train the face recognizer. We will do that by calling the `train(faces-vector, labels-vector)` method of face recognizer.


# Training of recognizer with training data(extracted faces and labels from images folder)
# Numpy is the fundamental package for scientific computing with Python. It creates powerful N-dimensional array object.
face_recognizer.train(faces, np.array(labels))


# ### Recognize ###

# function to draw rectangle on detected face
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


# function to draw person name
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


# recognize and name it
def recognize(test_img):
    img = test_img.copy()
    # detect face from the image
    face, rect = detect_face(img)

    # recognize
    label, confidence = face_recognizer.predict(face)
    # find related label
    label_text = subjects[label]

    # draw rectangle around face
    draw_rectangle(img, rect)
    # write the name
    draw_text(img, label_text, rect[0], rect[1] - 5)

    return img


print("Recognizing...")

test_images = os.listdir("test-data")
for test_image_name in test_images:
    if not test_image_name.startswith("test"):
        continue
    test_image_path = "test-data" + "/" + test_image_name
    image = cv2.imread(test_image_path)

    # perform recognition
    predicted_img = recognize(image)
    # display images
    cv2.imshow("Result for " + test_image_name, cv2.resize(predicted_img, (400, 500)))


print("Recognition complete")
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
