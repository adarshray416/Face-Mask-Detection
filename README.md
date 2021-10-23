# Face-Mask-Detection
One time Setup:

1.	Download https://github.com/tensorflow/models and
 save it 

2. Next, open anaconda (I am using anaconda as it provides a one stop place)
We can use anaconda directly but it is advised to create a virtual environment and use it.

create virtual environment named "object" and then activate it

=>>  conda create -n object

=>>   conda activate object

Verify version   =>>  python --version


To deactivate a environment 
=>>  conda deactivate 

To delete a environment  
=>> conda env remove -n object

To delete cache
=>> conda clean --all


3.	Install python for this virtual environment

     =>>  conda install python=3.7
  Verify version   =>>  python --version

4.   Download Protocol Buffer this is used for string parsing, etc and it is one of most important step.
Download this github (Scroll to end)  and download your specific windows or linux version for windows download  protoc-3.4.0-win32.zip:   https://github.com/protocolbuffers/protobuf/releases/tag/v3.4.0

Extract this and then goto bin folder then copy the protoc.exe and paste it into step 1 downloaded models/research  folder

5.  cd Folder_name/models/research/
6.Compile the protoc just downloaded 
=>>    protoc object_detection/protos/*.proto --python_out=.

7. Before proceeding,
=>>   pip install TensorFlow==1.15    

I am downloading tensorflow 1.15 because we need any tensorflow 1.x.x
 On Windows, you must have the Visual C++ 2015 build tools on your path. If you don't, make sure to install them from here:  

8. Run to build and install all required packages 
Copy  setup.py from  models\research\object_detection\packages\tf2 and paste it into  models\research  
=>>    python -m pip install .

**Training Steps**

1.	**Data Acquiring**
Collect data from any open source dataset repositories like Kaggle or Robocon

a.	Here I have created my own dataset by clicking images or downloading them individually from google.

b.	If you have created your custom data we need to make some changes accordingly. (Label Images)

To label image I am using LabelImage but we have other alternative tools too (VGG Image Annotation Tool and VoTT (Visual Object Tagging Tool).
INSTALL Labelimage and open it from the instructions given in readme.
Labelimage saves the file in XML Format

2.	**Convert Xml file into CSV**

Divide all the images into two parts (Test and Train). Keep the image in 80 : 20 Ratio and 
move 80% Images into Train Folder whereas 20% Into Test Folder.

Run the file xml_to_csv by following command
**python xml_to_csv.py**

3.	**Convert CSV Single file into TFRecord**

TWO csv files (Train & Test) would have been created after executing the above file.
To convert the CSV file into record file run the generate_tfrecord.py 

**Change the following section in the file**
def class_text_to_int(row_label):
    if row_label == 'with_mask':
        return 1
    elif row_label == 'without_mask':
        return 2
    elif row_label == 'mask_weared_incorrect':
        return 3
    else:
        return None

**	python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record**

**	python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record**


**NOW WE ARE READY FOR TRAINING**
Download and save the label_map file which will be used in below steps.

Goto models/ research/object_detection/configs/tf2

You can use any of the model configuration file. I will be using Efficientdet but there are other more available models in TensorFlow 2 Object Detection model zoo.
Download the model and store it from TensorFlow 2 Object Detection model zoo.  (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

Open the configuration file and change following details:
1.	Change Num_classes = 3 as I have used 3 labels.
2.	Change fine_tune_checkpoint to the path of downloades_model/checkpoint/ckpt-0
3.	Change fine_tune_checkpoint_type to detection
4.	Change batch size to 8 or 12 or 16 or any number depending on you system capability.
5.	Change input_path of the train_input_reader to the path of the train.record file
6.	Change input_path of the eval_input_reader to the path of the test.record
7.	Change label_map_path of train_input as well as eval_input to the path of the label map.

Now we can run the training command inside model/research/object_detection

**python model_main_tf2.py \ --pipeline_config_path=training/ssd_efficientdet_d0_512x512_coco17_tpu-8.config \ --model_dir=training \ --alsologtostderr**

When the loss is significant enough low we can stop the training


**Export inference graph**

Once the training is done to a satisfaction level run the below code. This will create a inference graph directory which will be used for final test and execution.

**python exporter_main_v2.py \ --trained_checkpoint_dir=training \ --pipeline_config_path=training/ssd_efficientdet_d0_512x512_coco17_tpu-8.config \ --output_directory inference_graph**

Now we are ready to execute and verify our trained model.
USE THE FILE run_py and make changes in path of test images and inference graph. 
Or you can use the file run1_py for live streaming object detection.

