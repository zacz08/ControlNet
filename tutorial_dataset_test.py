from tutorial_dataset_bev import MyDataset

dataset = MyDataset()
print(len(dataset))

item = dataset[5]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
print(txt)
print(jpg.shape)
print(hint.shape)
