import torch
from tokenizers import Tokenizer


tokenizer = Tokenizer.from_file('tokenize_path_here')


class MLMDataset:
    def __init__(self,lines):
        self.lines = lines
    def __len__(self,):
        return len(self.lines)
    def __getitem__(self,idx):
        line = self.lines[idx]
        ids = tokenizer.encode(line).ids
        labels = ids.copy()
        return ids, labels
    

def collate_fn(batch):
    input_ids = [torch.tensor(i[0]) for i in batch]
    labels = [torch.tensor(i[1]) for i in batch]
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)
    # mask 15% of text leaving [PAD]
    mlm_mask = torch.rand(input_ids.size()) < 0.15 * (input_ids!=1)
    masked_tokens = input_ids * mlm_mask
    labels[masked_tokens==0]=-100 # set all tokens except masked tokens to -100
    input_ids[masked_tokens!=0]=2 # MASK TOKEN
    return input_ids, labels