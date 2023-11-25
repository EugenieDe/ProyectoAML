import json
import os
import os.path as osp
import random

def gen_split(train_path, val_path, split_path):
    endovis2018_train_path = train_path
    endovis2018_im_train_path = osp.join(endovis2018_train_path, 'images')
    train_names = os.listdir(endovis2018_im_train_path)

    endovis2018_val_path = val_path
    endovis2018_im_val_path = osp.join(endovis2018_val_path, 'images')
    val_names = os.listdir(endovis2018_im_val_path)

    train_names.sort()
    val_names.sort() 

    train_names_complete = []
    val_names_complete = []

    for name in train_names:
        image = name
        name = osp.join(endovis2018_im_train_path, name)
        endovis2018_mask_train_path = osp.join(osp.join(endovis2018_train_path, 'annotations'), image)
        name = name + ' ' + endovis2018_mask_train_path + '\n'
        train_names_complete.append(name)

    for name in val_names:
        image = name
        name = osp.join(endovis2018_im_val_path, name)
        endovis2018_mask_val_path = osp.join(osp.join(endovis2018_val_path, 'annotations'), image)
        name = name + ' ' + endovis2018_mask_val_path + '\n'
        val_names_complete.append(name)

    train_names_complete = sorted(train_names_complete, key=lambda x: random.random())

    split_path = split_path

    split_train = open(osp.join(split_path, 'train.txt'), 'x')
    split_val = open(osp.join(split_path, 'val.txt'), 'x')

    split_train.writelines(train_names_complete)
    split_val.writelines(val_names_complete)

    split_train.close()
    split_val.close()


def split(num, split_path, train_path):   
    split_path = split_path + str(num)

    endovis2018_train_path = train_path
    endovis2018_im_train_path = osp.join(endovis2018_train_path, 'images')
    train_names = os.listdir(endovis2018_im_train_path)

    train_names.sort()

    labeled = []
    unlabeled = []

    for i in range(0,len(train_names)):
        name = train_names[i]
        image = name
        name = osp.join(endovis2018_im_train_path, name)
        endovis2018_mask_train_path = osp.join(osp.join(endovis2018_train_path, 'annotations'), image)
        name = name + ' ' + endovis2018_mask_train_path + '\n'
        if i%num == 0:
            unlabeled.append(name)
            #labeled.append(name)
        else:
            #unlabeled.append(name)
            labeled.append(name)

    labeled = sorted(labeled, key=lambda x: random.random())
    unlabeled = sorted(unlabeled, key=lambda x: random.random())

    split_labeled = open(osp.join(split_path, 'labeled.txt'), 'x')
    split_unlabeled = open(osp.join(split_path, 'unlabeled.txt'), 'x')

    split_labeled.writelines(labeled)
    split_unlabeled.writelines(unlabeled)

    split_labeled.close()
    split_unlabeled.close()



def create_json_splits(train_json_path, val_json_path, labeled_path, unlabeled_path, output_labeled_path, output_unlabeled_path):

    endo_train_json_path = train_json_path 
    endo_val_json_path = val_json_path

    endo_splits_labeled_path = labeled_path
    endo_splits_unlabeled_path = unlabeled_path
    endo_train = json.load(open(endo_train_json_path))

    with open(endo_splits_labeled_path, 'r') as f:
        id_labeled = f.read().splitlines()
    with open(endo_splits_unlabeled_path, 'r') as f:
        id_unlabeled = f.read().splitlines()
    
    list_name_im_labeled = []
    list_name_im_unlabeled = []

    for i in range(0, len(id_labeled)):
        im_name = id_labeled[i].split("/")[-1]
        list_name_im_labeled.append(im_name)

    for i in range(0, len(id_unlabeled)):
        im_name = id_unlabeled[i].split("/")[-1]
        list_name_im_unlabeled.append(im_name)

    dict_endo_train_labeled = {}
    dict_endo_train_labeled['info'] = endo_train['info']
    dict_endo_train_labeled['licenses'] = endo_train['licenses']
    dict_endo_train_labeled['categories'] = endo_train['categories']
    dict_endo_train_labeled['images'] = []
    dict_endo_train_labeled['annotations'] = []

    dict_endo_train_unlabeled = {}
    dict_endo_train_unlabeled['info'] = endo_train['info']
    dict_endo_train_unlabeled['licenses'] = endo_train['licenses']
    dict_endo_train_unlabeled['categories'] = endo_train['categories']
    dict_endo_train_unlabeled['images'] = []
    dict_endo_train_unlabeled['annotations'] = []

    list_im_id_labeled=[]
    list_im_id_unlabeled=[]
    j=0
    k=0
    for i in range(0, len(endo_train['images'])):
        im_name = endo_train['images'][i]['file_name']
        if im_name in list_name_im_labeled:
            dict_to_append = endo_train['images'][i]
            dict_to_append['new_id'] = j
            dict_endo_train_labeled['images'].append(dict_to_append)
            list_im_id_labeled.append(endo_train['images'][i]['id'])
            j+=1
        else:
            dict_to_append = endo_train['images'][i]
            dict_to_append['new_id'] = k
            dict_endo_train_unlabeled['images'].append(dict_to_append)
            list_im_id_unlabeled.append(endo_train['images'][i]['id'])
            k+=1
        
    k=0
    for i in range(0, len(endo_train['annotations'])):
        im_id = endo_train['annotations'][i]['image_id']
        if im_id in list_im_id_labeled:
            dict_to_append = endo_train['annotations'][i]
            dict_to_append['new_id'] = k
            dict_endo_train_labeled['annotations'].append(dict_to_append)
            k+=1


    with open(output_labeled_path, 'w') as outfile:
        json.dump(dict_endo_train_labeled, outfile)
    with open(output_unlabeled_path, 'w') as outfile:
        json.dump(dict_endo_train_unlabeled, outfile)

def main():
    train_path = "/home/eugenie/These/data/endovis2018/train"
    val_path = "/home/eugenie/These/data/endovis2018/val"
    gen_split_path = "/home/eugenie/These/UniMatch/splits/endovis2018/unsorted"
    split_path = "/home/eugenie/These/UniMatch/splits/endovis2018/unsorted/1_"

    train_json_path = "/home/eugenie/EndoVis/data/endovis2018/RobotSeg2018_inst_class_train.json"
    val_json_path = "/home/eugenie/EndoVis/data/endovis2018/RobotSeg2018_inst_class_val.json"

    labeled_path_1_2 = "/home/eugenie/EndoVis/proyectoaml202320/unimatch/splits/endovis2018/unsorted/1_2/labeled.txt"
    unlabeled_path_1_2 = "/home/eugenie/EndoVis/proyectoaml202320/unimatch/splits/endovis2018/unsorted/1_2/unlabeled.txt"
    output_labeled_path_1_2 = "/home/eugenie/EndoVis/data/endovis2018/train/splits/1_2/labeled.json"
    output_unlabeled_path_1_2 = "/home/eugenie/EndoVis/data/endovis2018/train/splits/1_2/unlabeled.json"

    labeled_path_1_4 = "/home/eugenie/EndoVis/proyectoaml202320/unimatch/splits/endovis2018/unsorted/1_4/labeled.txt"
    unlabeled_path_1_4 = "/home/eugenie/EndoVis/proyectoaml202320/unimatch/splits/endovis2018/unsorted/1_4/unlabeled.txt"
    output_labeled_path_1_4 = "/home/eugenie/EndoVis/data/endovis2018/train/splits/1_4/labeled.json"
    output_unlabeled_path_1_4 = "/home/eugenie/EndoVis/data/endovis2018/train/splits/1_4/unlabeled.json"

    labeled_path_1_8 = "/home/eugenie/EndoVis/proyectoaml202320/unimatch/splits/endovis2018/unsorted/1_8/labeled.txt"
    unlabeled_path_1_8 = "/home/eugenie/EndoVis/proyectoaml202320/unimatch/splits/endovis2018/unsorted/1_8/unlabeled.txt"
    output_labeled_path_1_8 = "/home/eugenie/EndoVis/data/endovis2018/train/splits/1_8/labeled.json"
    output_unlabeled_path_1_8 = "/home/eugenie/EndoVis/data/endovis2018/train/splits/1_8/unlabeled.json"

    gen_split(train_path, val_path, gen_split_path)
    split(2, split_path, train_path)
    split(4, split_path, train_path)
    split(8, split_path, train_path)

    create_json_splits(train_json_path, val_json_path, labeled_path_1_2, unlabeled_path_1_2, output_labeled_path_1_2, output_unlabeled_path_1_2)
    create_json_splits(train_json_path, val_json_path, labeled_path_1_4, unlabeled_path_1_4, output_labeled_path_1_4, output_unlabeled_path_1_4)
    create_json_splits(train_json_path, val_json_path, labeled_path_1_8, unlabeled_path_1_8, output_labeled_path_1_8, output_unlabeled_path_1_8)

    
