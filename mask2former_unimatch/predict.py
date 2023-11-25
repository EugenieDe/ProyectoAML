import sys
sys.path.insert(0, "Mask2Former")
import tempfile
from pathlib import Path
import numpy as np
import cv2
#import cog.python.cog

# import some common detectron2 utilities
from detectron2.config import CfgNode as CN
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2_git.projects.DeepLab.deeplab import add_deeplab_config

# import Mask2Former project
from mask2former import add_maskformer2_config

json_path_unlabeled = "/home/eugenie/These/data/endovis2018/train/splits/1_2/unlabeled.json"
json_path_labeled = "/home/eugenie/These/data/endovis2018/train/splits/1_2/labeled.json"
json_path_val = "/home/eugenie/These/data/endovis2018/val/RobotSeg2018_inst_class_val.json"

from detectron2.data.datasets import register_coco_instances

metadata = {'evaluator_type': 'sem_seg',}

register_coco_instances('endovis2018_train_1_2_unlabeled', {}, json_path_unlabeled, "/home/eugenie/These/data/endovis2018/train/images")
register_coco_instances('endovis2018_train_1_2_labeled', {}, json_path_labeled, "/home/eugenie/These/data/endovis2018/train/images")
register_coco_instances('endovis2018_val', {}, json_path_val, "/home/eugenie/These/data/endovis2018/val/images")

class Predictor():
    def setup(self):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file("/home/eugenie/These/proyectoaml202320/mask2former_unimatch/configs/endovis2018/SwinS.yaml")
        cfg.MODEL.WEIGHTS = '/home/eugenie/These/proyectoaml202320/mask2former_unimatch/output/endovis2018/SwinS_base_test0/model_final.pth'
        cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
        cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
        cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
        self.predictor = DefaultPredictor(cfg)
        self.coco_metadata = MetadataCatalog.get("endovis2018_val")

    def predict(self, image):
        im = cv2.imread(str(image))
        outputs = self.predictor(im)
        #v = Visualizer(im[:, :, ::-1], self.coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        #panoptic_result = v.draw_panoptic_seg(outputs["panoptic_seg"][0].to("cpu"),
        #                                     outputs["panoptic_seg"][1]).get_image()
        v = Visualizer(im[:, :, ::-1], self.coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        instance_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
        #v = Visualizer(im[:, :, ::-1], self.coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        #semantic_result = v.draw_sem_seg(outputs["sem_seg"].argmax(0).to("cpu")).get_image()
        #result = np.concatenate((panoptic_result, instance_result, semantic_result), axis=0)[:, :, ::-1]
        out_path = "/home/eugenie/These/proyectoaml202320/mask2former_unimatch/output/endovis2018/SwinS_base_test0/images/out.png"
        cv2.imwrite(str(out_path), instance_result)
        return out_path

predictor = Predictor()
predictor.predict("/home/eugenie/These/data/endovis2018/val/images/seq_2_frame051.png")
