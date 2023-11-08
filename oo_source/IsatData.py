import os
import pandas as pd
import xarray as xr

class IsatData():
    def __init__(self, shot_settings_path, check=False):
        r"""
        Initialize, read and store shotlog.
        """
        if check and os.path.isFile(shot_settings_path):
            raise FileNotFoundError(shote_settings_path)
        self.shot_settings_path = shotlog_path
        compressed_shot_settings = pd.read_excel(shot_file, skiprows=2, header=0)
        self.shot_settings = expand_shot_settings(compressed_shot_settings)
        self._shots = None
        
        
    def expand_shot_settings(compressed_shot_settings):
        r"""
        
        """
        shot_settings = []
        for i, shot in enumerate(compressed_shot_settings['Shots']): 
            if str(shot) == 'nan':
                continue
            if '-' not in shot:
                shot_settings.append(compressed_shot_settings.iloc[i].to_dict())
            if '-' in shot:
                shot_start, shot_end = shot.split('-')
                date = shot_start[:-3]
                start = shot_start[-3:]
                for j in range(int(start), int(shot_end) + 1):
                    shot_to_save = date + str(j).zfill(3)
                    settings = compressed_shot_settings.iloc[i].to_dict()
                    settings['Shots'] = shot_to_save
                    shot_settings.append(settings)
        shot_settings = pd.DataFrame(shot_settings)
        
        
    def isats(self, shot):
        
        
        
    def read_mdsp(shot, server):
        r"""
        
        """
        
        
    def map():
        
        
    def plot_together(axes=None):
    
    
    def apply_rj():
        
        
    def 