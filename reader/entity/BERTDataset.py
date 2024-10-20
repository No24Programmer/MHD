from torch.utils.data import Dataset

class BERTDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data[0])

    """
    0: input_ids    
    1: input_mask
    2: mask_position
    3: mask_label
    4: mask_type   
    5: attention_mask
    6: mask_token_index
    7: token_ids
    8: labels
    """
    def __getitem__(self, index):
        return self.data[0][index], \
               self.data[1][index], \
               self.data[2][index], \
               self.data[3][index], \
               self.data[4][index], \
               self.data[5][index], \
               self.data[6][index], \
               self.data[7][index], \
               self.data[8][index]

