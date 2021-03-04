from torch.utils.data import DataLoader,Dataset
import pandas as pd

class BuildDataSet(Dataset):
    def __init__(self,dataset):
        self.dataset = dataset.values
        self._len = len(self.dataset)

    def __getitem__(self, index):
        example=self.dataset[index]
        return [example]

    def __len__(self):
        return self._len

df=pd.read_csv('./data/Preliminary/gaiic_track3_round1_testA_20210228.tsv')
data=BuildDataSet(df)
load_data=DataLoader(dataset=data,batch_size=5,shuffle=True)

for batch,data in enumerate(load_data):
    print(batch)
