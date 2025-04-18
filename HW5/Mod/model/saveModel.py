#!/usr/bin/env python
# coding: utf-8

import numpy as np
import config
import tensorflow as tf

def save_model(model, model_name=None):
    if model_name is None:
        model_file_name = "kws_model.h5"
    else:
        model_file_name = model_name
        
    print(f"Saving model to {model_file_name}")
    model.save(model_file_name, overwrite=True)
    
    return model_file_name

def save_model_info(model_file_name, test_acc, tpr, fpr, spectrogram_shape):
    info_file_name = model_file_name.split('.')[0] + '.txt'
    label_list = config.get_label_list()
    
    with open(info_file_name, 'w') as fpo:
        fpo.write(f"i16min            = {config.I16MIN}\n")
        fpo.write(f"i16max            = {config.I16MAX}\n")
        fpo.write(f"fsamp             = {config.FSAMP}\n")
        fpo.write(f"wave_length_ms    = {config.WAVE_LENGTH_MS}\n")
        fpo.write(f"wave_length_samps = {config.WAVE_LENGTH_SAMPS}\n")
        fpo.write(f"window_size_ms    = {config.WINDOW_SIZE_MS}\n")
        fpo.write(f"window_step_ms    = {config.WINDOW_STEP_MS}\n")
        fpo.write(f"num_filters       = {config.NUM_FILTERS}\n")
        fpo.write(f"use_microfrontend = {config.USE_MICROFRONTEND}\n")
        fpo.write(f"learning_rate     = {config.LEARNING_RATE}\n")
        fpo.write(f"label_list        = {label_list}\n")
        fpo.write(f"spectrogram_shape = {spectrogram_shape}\n")
        fpo.write(f"Test set accuracy =  {test_acc:.1%}\n")
        
        for i in range(len(label_list)):
            fpo.write(f"tpr_{label_list[i]:9} = {tpr[i]:.3f}\n")
            fpo.write(f"fpr_{label_list[i]:9} = {fpr[i]:.3f}\n")
    
    print(f"Wrote description to {info_file_name}")
    return info_file_name
