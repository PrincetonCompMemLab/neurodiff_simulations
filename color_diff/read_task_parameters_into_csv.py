
# coding: utf-8

# In[18]:

import csv
import pandas as pd
import numpy as np

import os as os
import sys as sys
from collections import defaultdict
import time
from IPython.display import display
import copy

import traceback


# In[2]:

from datetime import datetime

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Current Time1 =", current_time)


# In[3]:

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))
print('---')
from_cmdLine = sys.argv[-1]
print(from_cmdLine)

print('---', flush=True)


# In[10]:

if from_cmdLine == 'cmd' :
    data_file = sys.argv[-2]
    print('using command line to set data')
    dataDir = data_file + '/'
else :
    data_file = '2022-01-07-11-59-16'
    dataDir = 'data/' + data_file + '/'


# In[11]:

dataDir


# In[12]:

# Set dimensions of hidden layer
hidden_dimensions = 1

# Set tasks being run
color_task_run = os.path.exists(dataDir + 'color_diff_Base_Params_TaskColorRecall.csv')
face_task_run = os.path.exists(dataDir + 'color_diff_Base_Params_TaskFaceRecall.csv')


if color_task_run and not face_task_run :
    task_run = 'just_color'
elif color_task_run and face_task_run :
    task_run = 'interleaved'
elif not color_task_run and face_task_run :
    task_run = 'just_face'
    
else:
    task_run = 0
    raise ValueError('something went wrong. Task not run properly.')

    


# In[13]:

def get_parameter_values():
    if task_run != 'just_face':
        params_file = dataDir + "color_diff_Base_Params_TaskColorRecall.csv"
    else: 
        params_file = dataDir + "color_diff_Base_Params_TaskFaceRecall.csv"
        
    Layer_ThrP_NMPH_dict = {}
    Layer_Drev_dict = {}
    Layer_DThr_dict = {}
    LTD_mult_TaskColorRecall = None
    Layer_OscAmnt_dict = {}
    Layer_Gi_dict = {}
    overlap_dict = {}
    
    parameter_values = {}

    with open(params_file, 'r') as f:
        for line in f.readlines():
            if line[:6] == "Layer:": # get layer name
                layername = line.split(" ")[-1][:-1]

            if line[:5] == "Prjn:": # get layer name
                prjnname = line.split(" ")[-1][:-1]
                
            if "OscAmnt" in line: # get oscAmnt
                Layer_OscAmnt_dict[layername] = float(line.split(" ")[-2])
            if "Layer" in line and "Gi" in line:
                Layer_Gi_dict[layername] = float(line.split(" ")[9])
                
            if "ThrP_NMPH" in line: # get AveL value (only occurs 4 times in csv file)
                Layer_ThrP_NMPH_dict[prjnname] = float(line.split(" ")[-8])
            if "DRev_NMPH" in line: # get AveL value (only occurs 4 times in csv file)
                Layer_Drev_dict[prjnname] = float(line.split(" ")[-20])
            if (LTD_mult_TaskColorRecall == None) and ('LTD_mult' in line):
                LTD_mult_TaskColorRecall = float(line.split(" ")[-2])
            if "DThr_NMPH" in line:
                Layer_DThr_dict[prjnname] = float(line.split(" ")[-23])

    if task_run != 'just_face':
        params_file = dataDir + "color_diff_Base_Params_TaskColorRecall.csv"
    else: 
        params_file = dataDir + "color_diff_Base_Params_TaskFaceRecall.csv"
        
    Color_to_Hidden_wt_scale = {}
    with open(params_file, 'r') as f:
        for line in f.readlines():
            if line[:5] == "Prjn:": # get layer name
                prjnname = line.split(" ")[-1][:-1]
            if "Rel:" in line: # get AveL value (only occurs 4 times in csv file)
                Color_to_Hidden_wt_scale[prjnname] = float(line.split(" ")[-1])
            if "NumOverlapUnits" in line:
                
                line_split = line.split(" ")
                overlap_dict['numOverlapUnits'] = int(line_split[[i for i, x in enumerate(line_split) if "NumOverlapUnits" in x][0] + 1])
                overlap_dict['numTotalUnits'] = int(line_split[[i for i, x in enumerate(line_split) if "NumTotalUnits" in x][0] + 1])
                overlap_dict['overlapType'] = str(overlap_dict['numOverlapUnits']) + '/' + str(overlap_dict['numTotalUnits'])



                #### OLD VERSION, BEFORE ADDING SPARSE POSSIBILITY. 
#                 overlap_dict['numOverlapUnits'] = int(line.split(" ")[-3])
#                 overlap_dict['numTotalUnits'] = int(line.split(" ")[-1])
#                 overlap_dict['overlapType'] = str(overlap_dict['numOverlapUnits']) + '/' + str(overlap_dict['numTotalUnits'])
        
    parameter_values['ThrP_NMPH'] = Layer_ThrP_NMPH_dict
    parameter_values['DRev_NMPH'] = Layer_Drev_dict
    parameter_values['DThr_NMPH'] = Layer_DThr_dict

    parameter_values['LTD_mult'] = LTD_mult_TaskColorRecall
    parameter_values['Wt_Scale'] = Color_to_Hidden_wt_scale
    parameter_values['OscAmnt'] = Layer_OscAmnt_dict
    parameter_values['Gi'] = Layer_Gi_dict
    
    parameter_values['Num_units_per_layer'] = {
        'Object': 2, 'Face': 6, 'Output': 50
    }
    if hidden_dimensions == 1:
        parameter_values['Num_units_per_layer']['Hidden'] = 50
    elif hidden_dimensions == 2:
        parameter_values['Num_units_per_layer']['Hidden'] = 100

    parameter_values['overlap'] = overlap_dict
    return parameter_values

# def get_parameter_values():
#     params_file = dataDir + "color_diff_Base_Params_TaskColorWOOsc.csv"
#     Color_to_Hidden_wt_scale = {}

#     with open(params_file, 'r') as f:
#         for line in f.readlines():
#             if line[:5] == "Prjn:": # get layer name
#                 prjnname = line.split(" ")[-1][:-1]
#             if "Rel:" in line: # get AveL value (only occurs 4 times in csv file)
#                 Color_to_Hidden_wt_scale[prjnname] = float(line.split(" ")[-1])

#     parameter_values = {}
#     parameter_values['Wt_Scale'] = Color_to_Hidden_wt_scale
#     return parameter_values

print('getting parameter values', flush=True)

parameter_values = get_parameter_values() 


# In[14]:

parameter_values


# In[15]:

def get_parameter_values(model_name = "Chanales"):
    if task_run != 'just_face':
        params_file = dataDir + "color_diff_Base_Params_TaskColorRecall.csv"
    else: 
        params_file = dataDir + "color_diff_Base_Params_TaskFaceRecall.csv"
        
    # Create a dictionary of Layer:{DICTIONARY of param values}
    Layer_Param_dict = defaultdict(dict)
    with open(params_file, 'r') as f:
        for line in f.readlines():
            if line[:6] == "Layer:": # get layer name
                layername = line.split(" ")[-1][:-1]
                
            if line[:5] == "Prjn:": # get layer name
                prjnname = line.split(" ")[-1][:-1]
                
            if "OscAmnt" in line: # get oscAmnt
                Layer_Param_dict[layername]["OscAmnt"] = float(line.split(" ")[-2])
                
            if "K_max" in line: # get K_max
                Layer_Param_dict[layername]["K_max"] = float(line.split(" ")[7])
                
            if "K_for_WTA" in line: # get K_max
                Layer_Param_dict[layername]["K_for_WTA"] = float(line.split(" ")[5])
                
            if "K_point" in line: # get K_max
                Layer_Param_dict[layername]["K_point"] = float(line.split(" ")[9])
                
            if "Target_diff" in line: # get K_max
                Layer_Param_dict[layername]["Target_diff"] = float(line.split(" ")[11])
                
            if "VmActThr" in line: # get K_max
                Layer_Param_dict[layername]["XX1_Gain"] = float(line.split(" ")[9])
                
            if "AvgGain" in line: # get K_max
                Layer_Param_dict[layername]["Clamp_Gain"] = float(line.split(" ")[22])
                
    # Create a dictionary of Param_values:{LIST of Layers}
    print("Layer_Param_dict", Layer_Param_dict)
    Param_Layer_dict = defaultdict(list)
    for layer, param_dict in Layer_Param_dict.items():
        Param_Layer_dict["Model"] = model_name
        if layer == "Face":
            layer = "Item"
        elif layer == "Object":
            layer = "Category"
        Param_Layer_dict["Layer"].append(layer)
        for param_name, param_value in param_dict.items():
            if param_name == "K_max" and param_value == -1:
                param_value = "+inf"
            elif param_name == "Clamp_Gain" and (layer == "Hidden" or layer == "Output"):
                param_value = "---"
            Param_Layer_dict[param_name].append(param_value)
    Param_Layer_df = pd.DataFrame.from_dict(Param_Layer_dict)
    cols = ['Model', 'Layer', 'K_for_WTA', 'K_max', 'K_point', 'Target_diff', 'OscAmnt', 'XX1_Gain', 'Clamp_Gain']
    Param_Layer_df = Param_Layer_df[cols]
    Param_Layer_df.to_csv(dataDir + "color_diff_Layer_Params.csv", index=False)
    return Param_Layer_df
        

print('getting parameter values', flush=True)

parameter_values = get_parameter_values() 
display(parameter_values)


# In[26]:

def get_parameter_values(model_name = "Chanales"):
    def replace_face_obj_by_item_category(layer):
        if layer == "Face":
            layer = "Item"
        elif layer == "Object":
            layer = "Category"
        return layer
    
    if task_run != 'just_face':
        params_file = dataDir + "color_diff_Base_Params_TaskColorRecall.csv"
    else: 
        params_file = dataDir + "color_diff_Base_Params_TaskFaceRecall.csv"
        
    # Create a dictionary of Layer:{DICTIONARY of param values}
    Prjn_Param_dict = defaultdict(dict)
    with open(params_file, 'r') as f:
        for line in f.readlines():
            if line[:6] == "Layer:": # get layer name
                layername = line.split(" ")[-1][:-1]
                
            if line[:5] == "Prjn:": # get layer name
                prjnname = line.split(" ")[-1][:-1]
                send, recv = [replace_face_obj_by_item_category(layer) for layer in prjnname.split("To")]
                prjnname = f"{send}To{recv}"
                
            if "DThr_NMPH" in line: # get DThr_NMPH
                idx = line.split(" ").index("DThr_NMPH:")
                Prjn_Param_dict[prjnname]["DThr_NMPH"] = float(line.split(" ")[idx + 1])
                
            if "DRev_NMPH" in line: # get DRev_NMPH
                idx = line.split(" ").index("DRev_NMPH:")
                Prjn_Param_dict[prjnname]["DRev_NMPH"] = float(line.split(" ")[idx + 1])
                
            if "DRevMag_NMPH" in line: # get DRevMag_NMPH
                idx = line.split(" ").index("DRevMag_NMPH:")
                Prjn_Param_dict[prjnname]["DRevMag_NMPH"] = float(line.split(" ")[idx + 1])

            if "ThrP_NMPH" in line: # get ThrP_NMPH
                idx = line.split(" ").index("ThrP_NMPH:")
                Prjn_Param_dict[prjnname]["ThrP_NMPH"] = float(line.split(" ")[idx + 1])
                
            if "DMaxMag_NMPH" in line: # get DMaxMag_NMPH
                idx = line.split(" ").index("DMaxMag_NMPH:")
                Prjn_Param_dict[prjnname]["DMaxMag_NMPH"] = float(line.split(" ")[idx + 1])
                
            if "Abs:" in line: # get wtscale
                Prjn_Param_dict[prjnname]["WtScale_Abs (forwards / backwards)"] = float(line.split(" ")[3])
                Prjn_Param_dict[prjnname]["WtScale_Rel (forwards / backwards)"] = float(line.split(" ")[5])
                
            if "InitStrategy" in line: # get wtscale
                mean = float(line.split(" ")[5])
                var = float(line.split(" ")[7])
                Prjn_Param_dict[prjnname]["Wt Range"] = f"{mean - var} - {mean + var}"
                
    # Create a dictionary of Param_values:{LIST of Layers}
    print("Prjn_Param_dict", Prjn_Param_dict)
    Layer_Order = {"Category": 0, "Item": 1, "Hidden": 2, "Output": 3}
    Param_Prjn_dict = defaultdict(list)
    for prjn, param_dict in Prjn_Param_dict.items():
        #print("prjn", prjn.split("To"))
        send, recv = [replace_face_obj_by_item_category(layer) for layer in prjn.split("To")]
        if Layer_Order[send] > Layer_Order[recv]:
            continue
        Param_Prjn_dict["Model"] = model_name
        Param_Prjn_dict["Projection"].append(prjn)
        for param_name, param_value in param_dict.items():
            if "(forwards / backwards)" in param_name:
                reverse_prjn_name = f"{recv}To{send}"
                param_value = str(param_value) + f"/{Prjn_Param_dict[reverse_prjn_name][param_name]}"
            Param_Prjn_dict[param_name].append(param_value)
    
    Param_Prjn_dict_table2 = {}
    for key, val in Param_Prjn_dict.items():
        if key in ["Model", "Projection"]:
            Param_Prjn_dict_table2[key] = val
        elif key not in ["WtScale_Abs (forwards / backwards)", "WtScale_Rel (forwards / backwards)", "Wt Range"]:
            Param_Prjn_dict_table2[key] = Param_Prjn_dict[key]
    
    Param_Prjn_dict = {key:val for key,val in Param_Prjn_dict.items() if key in ["Model", "Projection", "WtScale_Abs (forwards / backwards)", "WtScale_Rel (forwards / backwards)", "Wt Range"]}
    

    Param_Prjn_df_1 = pd.DataFrame.from_dict(Param_Prjn_dict)
    cols = Param_Prjn_df_1.columns.tolist() 
    Param_Prjn_df = Param_Prjn_df_1[cols]
    display(Param_Prjn_df_1)
    Param_Prjn_df_1.to_csv(dataDir + "color_diff_Prjn_Params_Table1.csv", index=False)
    
    Param_Prjn_df_2 = pd.DataFrame.from_dict(Param_Prjn_dict_table2)
    cols = Param_Prjn_df_2.columns.tolist() 
    Param_Prjn_df = Param_Prjn_df_2[cols]
    display(Param_Prjn_df_2)
    Param_Prjn_df_2.to_csv(dataDir + "color_diff_Prjn_Params_Table2.csv", index=False)
    return Param_Prjn_df_1, Param_Prjn_df_2

print('getting parameter values', flush=True)

parameter_values = get_parameter_values() 
# print(parameter_values)


# In[ ]:



