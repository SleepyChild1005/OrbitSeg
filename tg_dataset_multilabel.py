import torch.utils.data as data
import numpy as np
import os
from medpy.io import load

class TargetDataset(data.Dataset):
    def __init__(self, tg_data_dir, mode, fold_number=0, total_fold=4):
        self.tg_data_dir = tg_data_dir
        self.tg_seq_vol_dir = self.tg_data_dir + 'vol/'
        self.tg_seq_mask_dir = self.tg_data_dir + 'mask/'

        # axial only
        tg_patient_ids = np.array(
            ['Image_12', 'Image_13', 'Image_28',
             'Image_30', 'Image_31', 'Image_32', 'Image_36', 'Image_37', 'Image_39', 'Image_41',
             'Image_45', 'Image_47', 'Image_49', 'Image_51', 'Image_52', 'Image_53',
             'Image_55', 'Image_57', 'Image_59', 'Image_60', 'Image_62', 'Image_64', 'Image_65',
             'Image_66', 'Image_67', 'Image_68', 'Image_69', 'Image_70', 'Image_72'])

        tg_patient_ids = sorted(tg_patient_ids)
        
        tg_test_patient_ids = tg_patient_ids[fold_number::total_fold]
        if mode == "train":
            self.tg_mode_patient_ids = [i for i in tg_patient_ids if i not in tg_test_patient_ids]
        elif mode == "test":
            self.tg_mode_patient_ids = tg_test_patient_ids
        ###
        
        self.tg_seq_filename = self.load_filenames(self.tg_seq_vol_dir, self.tg_mode_patient_ids)

    def load_filenames(self, data_dir, mode_patient_id):
        target_filenames = []

        filenames = os.listdir(data_dir)
        for filename in filenames:
            patient_id = filename.split("__")[0]

            if patient_id in mode_patient_id:
                target_filenames.append(filename)
                
        return target_filenames

    def get_volume(self, data_dir):
        volume, _ = load(data_dir)
        volume = np.expand_dims(volume, axis=0)
        volume = np.transpose(volume, (3, 0, 1, 2))        
        return volume    ### [sequence, channel, x, y]
    
    def get_mask(self, data_dir):
        mask, _ = load(data_dir)
        mask = mask[:,:,1,:]
        mask = np.transpose(mask,(2,0,1))

        return mask      ### [channel, x, y]

    def get_num_patient(self):
        return len(self.tg_mode_patient_ids)

    def __len__(self):
        return len(self.tg_seq_filename)
    
    def __getitem__(self, index):
        vol_path = self.tg_seq_vol_dir + self.tg_seq_filename[index]
        mask_path = self.tg_seq_mask_dir + self.tg_seq_filename[index]

        vol = self.get_volume(vol_path)
        mask = self.get_mask(mask_path)

        return vol, mask


        
   