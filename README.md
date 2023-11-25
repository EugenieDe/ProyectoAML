# ProyectoAML
The dataset and checkpoints are on BCV002.

## Paths
-	Dataset: /home/eugenie/These/data/endovis2018
-	Checkpoints:
    -	Backbone (this will surely raise an error in [UniMatch/model/backbone/resnet.py](UniMatch/model/backbone/resnet.py) on line 152 while running the code as the pretrained weights could not be pushed to GitHub):
      	- /home/eugenie/These/ProyectoAML/Unimatch/pretrained
    -	Semi-supervised:
        -	/home/eugenie/These/UniMatch/exp/endovis2018/unimatch/base/r101_OHEM/1_2/seed
        -	/home/eugenie/These/UniMatch/exp/endovis2018/unimatch/base/r101_OHEM/1_4/seed
        -	/home/eugenie/These/UniMatch/exp/endovis2018/unimatch/base/r101_OHEM/1_8/seed
    -	Fully-supervised:
        -	/home/eugenie/These/UniMatch/exp/endovis2018/supervised/base/r101/1_2
        - /home/eugenie/These/UniMatch/exp/endovis2018/supervised/base/r101/1_4
        -	/home/eugenie/These/UniMatch/exp/endovis2018/supervised/base/r101/1_8
        -	/home/eugenie/These/UniMatch/exp/endovis2018/supervised/base/r101/all

## Installation
```bash
conda create --name mask2former python=3.8 -y
conda activate mask2former
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -U opencv-python

git clone https://github.com/EugenieDe/ProyectoAML.git
cd ProyectoAML
pip install -r requirements.txt

cd mask2former_unimatch/detectron2_git
pip install -e .
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git

cd ..
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

## Usage
To train the semi-supervised model on the dataset with 50% annotations, run:
```bash
python main.py --test --semi --split 1_2
```
To train the fully supervised model on the dataset with 50% annotations, run:
```bash
python main.py --test --fully --split 1_2
```
An image is chosen by default to test the model on. To test the model on this image, run:
```bash
python main.py --demo
```
Alternatively you can chose any image from the validation dataset.
To evaluate a checkpoint on the whoel validation dataset, you can use:
```bash
python UniMatch/evaluation.py
```

As commented in the paper, the best working model is our baseline which is the model implemented in main.py. Nonetheless, you will find the model we are currently working on in  [mask2former_unimatch](mask2former_unimatch). You can run it with [mask2former_unimatch/train_net.py](mask2former_unimatch/train_net.py) but it won’t yield conclusive results for the moment.
