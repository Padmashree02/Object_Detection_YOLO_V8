# Problem - Object Detection with Deep learning's YOLO V8 method.

Object detection using deep neural network are categories in two methods- 

  > One stage detector :- Effective detectors, that detects the object by analysing the whole image in a single pass.

  > Two stage detector :- Efficeint detectors good for its accuracy as these detectors detects the object first by gnerating region proposals and then on each proposed region object classification and bounding box are detected.

In this object detection part, the objects are detected by one stage method and using YOLO based model- YOLO V8.

YOLO V8 are the model that not only used for object detection application, but are also mostly used in instance segmentation (eg- use as an robot's visions) and image classification

Solution :-

  > Network/Layer - yolov8

  > Framework - PyTorch

  > Dataset (while training the network) - coco

  > Library - cv2, numpy, matplotlib.pyplot, ultralytics (aids to export YOLO V8 models)

  > Pipeline (Inderfence/testing) :-

      : Read the image.

      : Load YOLO V8 model from YOLO library. 
      
      : Export all YOLO models from .pt to .onnx format- as OpenCV DNN module requires the model to be in the .onnx format.
        Note- Exported models automatically downloads in the local or web drive

      : Load the exported model file.

      : Read the model's files using ONNX.
     
      : Load and extract the class names from the class file- coco dataset (80 classes).

      : Set the confidence threshold, nms threshold value- detects how well the object is detected in the detected bounding box. 
        Set the default image's height and width as per YOLO V8 model's architecture

      : Convert the image into blob format- helps the network to understand the image through blob format. 
        Note- the parameters of blob function to convert image into blob is as per the newtwork's configuration text file.
      
      : Set the blob image ready and pass the blob forward to the to the loaded model for detections.

      : The result of detection is list of list cardinally with detected (center_x, center_y) coordinate, width, height, confidence score of the box and then confidence score wrt other classes. 

      :  Before analyzing each detection, first step to transpose and squeeze the result of detection, so that it can match to an expented shape. 
      
      : For each detections, extract the classID, confidence score. 
      
      : Check whether the confidence score satisfies with the Confidence_threshold value. 
      
        - If yes, then extract the bounding box dimensions and normalise each dimension wrt to image's shape and stack the simensions in the box list (an empty defined list).

        - Else no, then will pass to the next detection.

      : Perform Non Maximum Suppression on each detected box which is stored in the box list- result the list of indices of those detection which are valid.
        Note- NMS aids to removes overlapped bounding boxes.
        
      : With respect to indices, draw the bounding box (by extracting the dimensions from box list respectively) with detected class name and cofidence score.
