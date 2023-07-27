'''
Author: Zheng Ma
Date: 2022-02-21 11:17:24
LastEditTime: 2022-02-21 11:17:24
LastEditors: Zheng Ma
Description: 
FilePath: /smiles_generate/utils/progressbar.py

'''
import progressbar



def get_progressbar(max_value):
    widgets = [
    progressbar.Percentage(), ' ',
    progressbar.Bar('*'), ' ',
    progressbar.Timer(), ' ',
    # progressbar.DynamicMessage('loss')
]
    bar = progressbar.ProgressBar(widgets=widgets, max_value=max_value)

    return bar