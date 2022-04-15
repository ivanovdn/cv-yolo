import torch
import yaml
import cv2
import glob
import os
import time
from torch import multiprocessing
from itertools import product
import sys

config_path = '/Users/dmytro.ivanov/Projects/cv/test/app/mnt/config/config.yaml'
config = yaml.safe_load(open(config_path))
ORCS_ClASS = config["ORCS_CLASS"]
EXTRA_PATH = '/Users/dmytro.ivanov/Projects/cv/test/'

def establish_model():
    model = torch.hub.load('ultralytics/yolov5',
                       'custom',
                       path=f'{EXTRA_PATH}{config["PATH_MODEL"]}/{config["MODEL"]}',
                       )
    model.conf = config["CONFIDENCE"]
    model.iou = 0.45
    return model


def image_inference(folder, model):
    files_list = glob.glob(f'{folder}/*')
    if files_list:
        for input_file in files_list:
            try:
                input_folder_name = input_file.split('/')[-2]
                input_file_name = input_file.split('/')[-1]
                name, file_format = input_file_name.split('.')
                res = model(input_file) 
                #return res.print()
                if ORCS_ClASS in set(res.pandas().xyxy[0]['class']):
                    res_pd = res.pandas().xyxy[0]
                    count_orcs = len(res_pd[res_pd['class'] == ORCS_ClASS])
                    confidence = "_".join([str(round(i, 2)) for i in res_pd[res_pd["class"] == ORCS_ClASS]["confidence"].values])
                    res.files = [f'{name}_{count_orcs}_{confidence}.jpg']
                    res.save(save_dir=f'{EXTRA_PATH}{config["PATH_OUTPUT"]}/{input_folder_name}/')
                os.remove(input_file)
            except:
                pass
        

if __name__ == '__main__':
    sys.path.insert(0, '/Users/dmytro.ivanov/.cache/torch/hub/ultralytics_yolov5_master')
    multiprocessing.set_start_method('spawn')
    model = establish_model()
    while True:
    # model.share_memory()
        folders_list = glob.glob(f'{EXTRA_PATH}{config["PATH_INPUT"]}/*')
        if folders_list:
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            pool.starmap(image_inference, list(product(folders_list, [model, ])))
            pool.close()
            pool.join()      
        time.sleep(1)