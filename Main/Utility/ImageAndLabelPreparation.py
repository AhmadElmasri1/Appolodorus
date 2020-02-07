import keras
import cv2
import numpy
import os

largeTrainingImageDirectory = 'RawImages/'
resizedTrainingImageDirectory = 'PreppedImages/'

def DataSetPreparation():
    listOfSubDirectories = os.listdir(largeTrainingImageDirectory)
    numberOfLabels = len(listOfSubDirectories)

    #Holds the name of the labels based on the folder name
    informationArray = []
    labelArray = []


    index = 0

    for subList in listOfSubDirectories:

        subDirectory = largeTrainingImageDirectory + subList + "/"
        newSubDirectory = resizedTrainingImageDirectory + subList + '/'
        os.mkdir(newSubDirectory)

        subDir = os.listdir(subDirectory)



        for fileName in subDir:

            imagePath = subDirectory + fileName
            print(imagePath)
            imageHolder = cv2.imread(imagePath)
            if(imageHolder.shape[0] < imageHolder.shape[1]):
                # imageHolder = cv2.rotate(imageHolder,cv2.ROTATE_90_CLOCKWISE)
                # cv2.imwrite(imagePath,imageHolder )
                # print('Image Rotated')
                preppedImagePath = newSubDirectory + fileName
                imageHolder = cv2.resize(imageHolder,(360,240))
                cv2.imwrite(preppedImagePath,imageHolder)
            else:
                preppedImagePath = newSubDirectory + fileName
                imageHolder = cv2.resize(imageHolder,(240,360))
                cv2.imwrite(preppedImagePath,imageHolder)
            label = [index,subList]
            print(label)
            labelArray.append(label)

        labelArray = numpy.asarray(labelArray)
        # informationArray.append(labelArray)
        index += 1

    labelArray = numpy.asarray(labelArray)
    print(labelArray.shape)
    print(labelArray)

    return labelArray

def LoadDataSet():
    listOfSubDirectories = os.listdir(largeTrainingImageDirectory)
    numberOfLabels = len(listOfSubDirectories)

    #Holds the name of the labels based on the folder name
    imageArray = []

    informationArray = []
    labelArray = []

    index = 0

    for subList in listOfSubDirectories:

        subDirectory = resizedTrainingImageDirectory + subList + '/'

        for fileName in os.listdir(subDirectory):
            imagePath = subDirectory + fileName
            print(imagePath)
            imageHolder = cv2.imread(imagePath)
            preppedImagePath = subDirectory + fileName

            imageHolder = cv2.imread(preppedImagePath)
            imageHolder = numpy.asarray(imageHolder)

            imageArray.append(imageHolder)
            label = numpy.array([[index,subList]])
            labelArray.append(label)


        # informationArray.append(labelArray)
    imageArray = numpy.asarray(imageArray)
    labelArray = numpy.asarray(labelArray)

    return (informationArray, imageArray)




DataSetPreparation()


# LoadDataSet()