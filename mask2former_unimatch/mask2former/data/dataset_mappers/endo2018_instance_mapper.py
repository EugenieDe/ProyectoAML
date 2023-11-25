# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging

import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Instances

from pycocotools import mask as coco_mask

__all__ = ["Endo2018InstanceDatasetMapper"]


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def build_transform_gen(cfg, is_train, mode):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"
    assert mode in ["train_l", "train_u", "val"], f"Mode is {mode}, should be either train_l, train_u or val"
    min_size_train = cfg.INPUT.MIN_SIZE_TRAIN
    max_size_train = cfg.INPUT.MAX_SIZE_TRAIN

    augmentation = []

    if mode=="train_l":
        augmentation.append(T.Resize((min_size_train,max_size_train)))
        augmentation.append(T.RandomFlip(prob=0.5, horizontal=True, vertical=False))
        augmentation.append(T.RandomApply(T.RandomRotation(angle=90), prob=0.5))
        augmentation.append(T.RandomFlip(prob=0.5, horizontal=False, vertical=True))

    elif mode=="val":
        augmentation.append(T.Resize((min_size_train,max_size_train)))

    elif mode=="train_u":
        augmentation.append(T.Resize((min_size_train,max_size_train)))
        augmentation.append(T.RandomFlip(prob=0.5, horizontal=True, vertical=False))
        augmentation.append(T.RandomApply(T.RandomRotation(angle=90), prob=0.5))
        augmentation.append(T.RandomFlip(prob=0.5, horizontal=False, vertical=True))
        augmentation.append(T.RandomApply(T.RandomContrast(intensity_min=0.5, intensity_max=1.5), prob=0.5))
        augmentation.append(T.RandomApply(T.RandomBrightness(intensity_min=0.5, intensity_max=1.5), prob=0.5))
        augmentation.append(T.RandomApply(T.RandomSaturation(intensity_min=0.5, intensity_max=1.5), prob=0.5))

    return augmentation


# This is specifically designed for the Endovis 2018 dataset.
class Endo2018InstanceDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        mode = None,
        *,
        tfm_gens_labeled,
        tfm_gens_unlabeled,
        tfm_gens_val,
        image_format,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.mode=mode
        self.tfm_gens_labeled = tfm_gens_labeled
        self.tfm_gens_unlabeled = tfm_gens_unlabeled
        self.tfm_gens_val = tfm_gens_val
        logging.getLogger(__name__).info(
            f"[Endo2018InstanceDatasetMapper] Full TransformGens used in training: {str(self.tfm_gens_labeled)}, {str(self.tfm_gens_unlabeled)}"
        )

        self.img_format = image_format
        self.is_train = is_train
    
    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens_labeled = build_transform_gen(cfg, is_train, mode="train_l")
        tfm_gens_unlabeled = build_transform_gen(cfg, is_train, mode="train_u")
        tfm_gens_val = build_transform_gen(cfg, is_train, mode="val")

        ret = {
            "is_train": is_train,
            "tfm_gens_labeled": tfm_gens_labeled,
            "tfm_gens_unlabeled": tfm_gens_unlabeled,
            "tfm_gens_val": tfm_gens_val,
            "image_format": cfg.INPUT.FORMAT,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
            mode (str): To apply the mapper to different datasets. Amongst 'train_l', 'train_u' and 'val'.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.mode in ["train_l", "train_u", "val"], f"Mode is {self.mode}, should be either train_l, train_u or val"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        padding_mask = np.ones(image.shape[:2])

        if self.mode == "train_l":
            image, transforms = T.apply_transform_gens(self.tfm_gens_labeled, image)
        elif self.mode == "train_u":
            image_u_s1, transforms = T.apply_transform_gens(self.tfm_gens_unlabeled, image)
            image_u_s2, transforms = T.apply_transform_gens(self.tfm_gens_unlabeled, image)
            image_u_w, transforms = T.apply_transform_gens(self.tfm_gens_labeled, image)
        elif self.mode == 'val' or self.is_train==False:
            image, transforms = T.apply_transform_gens(self.tfm_gens_val, image)

        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        if self.mode == 'train_u':
            dataset_dict["image_u_w"] = torch.as_tensor(np.ascontiguousarray(image_u_w.transpose(2, 0, 1)))
            dataset_dict["image_u_s1"] = torch.as_tensor(np.ascontiguousarray(image_u_s1.transpose(2, 0, 1)))
            dataset_dict["image_u_s2"] = torch.as_tensor(np.ascontiguousarray(image_u_s2.transpose(2, 0, 1)))
        else:
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        if not self.is_train or self.mode =='val':
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:

            if self.mode == 'train_u':
                dataset_dict["instances"] = None
                return dataset_dict

            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                # Let's always keep mask
                # if not self.mask_on:
                #     anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # NOTE: does not support BitMask due to augmentation
            # Current BitMask cannot handle empty objects
            instances = utils.annotations_to_instances(annos, image_shape)
            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            # Need to filter empty instances first (due to augmentation)
            instances = utils.filter_empty_instances(instances)
            # Generate masks from polygon
            h, w = instances.image_size
            # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                instances.gt_masks = gt_masks
            dataset_dict["instances"] = instances

        return dataset_dict
