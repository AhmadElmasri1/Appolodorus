from __future__ import absolute_import, division, print_function, unicode_literals

from imageai.Detection.Custom import DetectionModelTrainer
from imageai.Detection.Custom import CustomObjectDetection

import os
import imageai
import scipy
import cv2
import Camera
import numpy
# import tensorflow.include.tensorflow.




import tensorflow as tf
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
#     allow_soft_placement=True, log_device_placement=True))
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
holder = get_available_gpus()
print(holder)

# Creates a graph.
with tf.device('/device:XLA_GPU:0'):
    a = tf.constant([ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
# Creates a session with allow_soft_placement and log_device_placement set
# to True.
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
    allow_soft_placement=True, log_device_placement=True))
# Runs the op.
print(sess.run(c))

#
# # Run Image Training according to the labels and annotations in the training folder
# trainer = DetectionModelTrainer()
# trainer.setModelTypeAsYOLOv3()
# trainer.setDataDirectory(data_directory="../TrainingAndTestingImages/TrainingImages/PreparedTrainingImages/")
# # trainer.set
# trainer.setTrainConfig(object_names_array=["FinishedAssembly","FalseAssembly","Bolt","Nut","Washer"], batch_size=2, num_experiments=5,
#                        train_from_pretrained_model="../TrainingAndTestingImages/TrainingImages/PreparedTrainingImages/models/detection_model-ex-005--loss-0008.034.h5")
# trainer.trainModel()

detector = imageai.Detection.VideoObjectDetection()

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
# detector.detectObjectsFromImage()
detector.setModelPath("../TrainingAndTestingImages/TrainingImages/PreparedTrainingImages/models/detection_model-ex-005--loss-0006.378.h5")
detector.setJsonPath("../TrainingAndTestingImages/TrainingImages/PreparedTrainingImages/json/detection_config.json")
detector.loadModel()
#
mainCamera = Camera.CameraHandler()
mainCamera.start()

while True:
    colorImage = mainCamera.getColorFrame()
    depthImage = mainCamera.getDepthFrame()
    distanceImage = mainCamera.getDepthFrame()
    # if os.path.exists('/home/zema/Documents/Appolodorus/TrainingAndTestingImages/WebCamImage/CamImage.jpg'):
    #     os.remove('/home/zema/Documents/Appolodorus/TrainingAndTestingImages/WebCamImage/CamImage.jpg')
    cv2.imwrite("../TrainingAndTestingImages/WebCamImage/CamImage.jpg", colorImage)

    # videoPath = detector.detectCustomObjectsFromVideo(camera_input=camera,output_file_path='../Models/', frames_per_second=20, log_progress=True, minimum_percentage_probability=30)
    detectionResults = detector.detectObjectsFromImage(input_image="../TrainingAndTestingImages/WebCamImage/CamImage.jpg",
                                    output_image_path="../TrainingAndTestingImages/WebCamImage/Detection.jpg", output_type="array", minimum_percentage_probability=60)
    print(detectionResults[0].shape)


    print(detectionResults[-1])

    objects = detectionResults[-1]

    for object in objects:
        points = object.get('box_points')
        distanceSubset = distanceImage[points[1]:points[3],points[0]:points[2]]
        distanceToObject = numpy.mean(distanceSubset)

        labelText = object.get('name') + ' ' + str(distanceToObject)
        cv2.rectangle(depthImage,(points[0], points[1]), (points[2], points[3]), color=(255,255,255), thickness=2)
        cv2.putText(depthImage, labelText, (points[0], points[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
    # displayImage = cv2.imread("../TrainingAndTestingImages/WebCamImage/Detection.jpg")
    # displayImage = numpy.copy(detectionResults[0])
    cv2.imshow('detection', depthImage)
    cv2.waitKey(1)
    # if os.
    # os.remove('../TrainingAndTestingImages/WebCamImage/Detection.jpg')
    # for detection in detections:
    #     print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])


# #
