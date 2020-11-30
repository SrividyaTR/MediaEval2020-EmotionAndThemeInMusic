import os
import numpy as np
import pickle
from torch.utils import data


class AudioFolder(data.Dataset):
    


    def __init__(self, root, subset, tr_val='train', split=0):
        self.trval = tr_val
        self.root = root
        fn = '../../data/splits/split-%d/%s_%s_dict.pickle' % (split, subset, tr_val)
        self.get_dictionary(fn)

    def __getitem__(self, index):
        fn = os.path.join(self.root, self.dictionary[index]['path'][:-3]+'npy')
        full_spect = np.array(np.load(fn))
        spect = self.get_spectogram(full_spect, 1400)
        tags = self.dictionary[index]['tags']
        return spect.astype('float32'), tags.astype('float32'), self.dictionary[index]['path']

    def get_dictionary(self, fn):
        with open(fn, 'rb') as pf:
            dictionary = pickle.load(pf)
        self.dictionary = dictionary

    def __len__(self):
        return len(self.dictionary)
        
        
    def get_spectogram(self, full_spect, size=1400):
        full_len = full_spect.shape[1]
        if full_len > size:
            spect = full_spect[:, :size]
        else:
            diff = size-full_len
            spect = full_spect
            while(diff > 0):
                if diff>full_len:
                    spect = np.concatenate((spect,full_spect), axis=1)
                    diff = diff-full_len
                else:
                    spect = np.concatenate((spect, full_spect[:,:diff]), axis=1)
                    diff = 0                
        return spect


def get_audio_loader(root, subset, batch_size, tr_val='train', split=0, num_workers=0):
    shouldShuffle = True
    
    # no shuffling is done while loading test data so that prediction can be compared against groundtruth labels
    if tr_val == 'test':
        shouldShuffle = False
    data_loader = data.DataLoader(dataset=AudioFolder(root, subset, tr_val, split),
                                  batch_size=batch_size,
                                  shuffle=shouldShuffle,
                                  num_workers=num_workers)
    return data_loader

