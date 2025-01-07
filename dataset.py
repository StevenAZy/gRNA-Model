from torch.utils.data import Dataset, DataLoader

class PRDataset(Dataset):
    def __init__(self, file_path, transform=None):
        """
        :param file_path: 文本数据文件的路径
        :param transform: 可选的文本转换，例如文本预处理
        """
        self.file_path = file_path
        self.transform = transform
        self.samples = []

        with open(file_path, 'r') as file:
            for line in file:
                protein, rna = line.strip().split('\t')  # 假设文本和标签以制表符分隔
                self.samples.append((protein, rna))

    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)

    def __getitem__(self, idx):
        """获取指定索引的数据项"""
        protein, rna = self.samples[idx]
        
        if self.transform:
            protein = self.transform(protein)  # 可以对文本做一些预处理，例如 tokenization

        return protein, rna

# # 假设我们有一个文本文件，其中每行包含一个文本样本和标签
# file_path = 'dataset_test.txt'

# # 创建自定义文本数据集实例
# dataset = PRDataset(file_path=file_path)

# # 使用 DataLoader 加载数据，批量大小为 32
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# # 获取一个批次的数据
# for texts, labels in dataloader:
#     print(texts)
#     print(labels)
#     break  # 只获取一个批次
