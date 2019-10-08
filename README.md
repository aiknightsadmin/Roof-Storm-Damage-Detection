
SET UP THE MARK_RCNN PROJECT
1.	STEP ONE
Download the original project from the github: https://github.com/aiknightsadmin/Roof-Storm-Damage-Detection
INSTALLATION
--1.	Clone this repository：https://github.com/aiknightsadmin/Roof-Storm-Damage-Detection 
--2.	Install dependencies
pip3 install -r requirements.txt
--3.	Run setup from the repository root directory
python3 setup.py install
2.	CREATE .JSON FILE
Use this website to create .json file: http://www.robots.ox.ac.uk/~vgg/software/via/via.html
3.	SET UP YOUR OWN DATASETS
Create a folder in the root of the project named “dataset” and put your pictures and json files in the sub-folder “train” of the “datasets”
4.	TRAIN YOUR PROGRAME
To train your program to get the Forecasting Weights and Model 
Command：
#Train a new model starting from pre-trained COCO weights
python hail.py train --dataset=/home/.../mask_rcnn/data/hail/ --weights=coco

#Train a new model starting from pre-trained ImageNet weights
python3 hail.py train --dataset=/Users/yhuang24/Project/Data/Hail --weights=imagenet

#Continue training the last model you trained. This will find
#the last trained weights in the model directory.
python3 hail.py train --dataset=/Users/yhuang24/Project/Data/Hail --weights=last

5.	SELECT BEST MODEL
Using Tensorboard to do data visualization and select best case (best training step) as new Weights and Model
Tensorboard usage：
https://www.tensorflow.org/tensorboard/migrate
6.	USE “DETECT” FUNCTION TO PREDICT RESULTS
Using “detect” function to predict and show results of accuracy, and save predicted pictures in folder.
Command：
#Detect and color splash on a image with the last model you trained.
#This will find the last trained weights in the model directory.
python3 hailV3.py detect --weights=last --image=
Batch detection
python HailV3.py detect --dataset=dataset --weights=mask_rcnn_hail_0124.h5 --subset=dataset/train


