from torchvision.datasets import CIFAR100
import numpy as np
from PIL import Image


class iCIFAR100(CIFAR100):
    def __init__(self,root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None,
                 download=False):
        super(iCIFAR100,self).__init__(root,
                                       train=train,
                                       transform=transform,
                                       target_transform=target_transform,
                                       download=download)

        self.target_test_transform=target_test_transform
        self.test_transform=test_transform
        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []

    def concatenate(self,datas,labels):
        # Filter out empty arrays to avoid broadcast errors
        non_empty = [(d, l) for d, l in zip(datas, labels) if len(d) > 0]
        if len(non_empty) == 0:
            return np.array([]), np.array([])
        datas_filtered, labels_filtered = zip(*non_empty)
        con_data=datas_filtered[0]
        con_label=labels_filtered[0]
        for i in range(1,len(datas_filtered)):
            con_data=np.concatenate((con_data,datas_filtered[i]),axis=0)
            con_label=np.concatenate((con_label,labels_filtered[i]),axis=0)
        return con_data,con_label

    def getTestData(self, classes):
        datas,labels=[],[]
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        self.TestData, self.TestLabels=self.concatenate(datas,labels)

    def getTrainData(self, classes, exemplar_set, exemplar_label_set):
        datas,labels=[],[]
        if len(exemplar_set)!=0 and len(exemplar_label_set)!=0:
            datas=[exemplar for exemplar in exemplar_set]
            length=len(datas[0])
            labels=[np.full((length), label) for label in exemplar_label_set]

        for label in classes:
            data=self.data[np.array(self.targets)==label]
            datas.append(data)
            labels.append(np.full((data.shape[0]),label))
        self.TrainData, self.TrainLabels=self.concatenate(datas,labels)

    def getSampleData(self, classes, exemplar_set, exemplar_label_set, group):
        datas,labels=[],[]
        if len(exemplar_set)!=0 and len(exemplar_label_set)!=0:
            datas=[exemplar for exemplar in exemplar_set]
            length=len(datas[0])
            labels=[np.full((length), label) for label in exemplar_label_set]

        if group == 0:
            for label in classes:
                data=self.data[np.array(self.targets)==label]
                datas.append(data)
                labels.append(np.full((data.shape[0]),label))
        self.TrainData, self.TrainLabels=self.concatenate(datas,labels)

    def getTrainItem(self,index):
        img, target = Image.fromarray(self.TrainData[index]), self.TrainLabels[index]

        if self.transform:
            img=self.transform(img)

        if self.target_transform:
            target=self.target_transform(target)

        return index,img,target

    def getTestItem(self,index):
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]

        if self.test_transform:
            img=self.test_transform(img)

        if self.target_test_transform:
            target=self.target_test_transform(target)

        return index, img, target

    def __getitem__(self, index):
        if len(self.TrainData) > 0:
            return self.getTrainItem(index)
        elif len(self.TestData) > 0:
            return self.getTestItem(index)


    def __len__(self):
        if len(self.TrainData) > 0:
            return len(self.TrainData)
        elif len(self.TestData) > 0:
            return len(self.TestData)

    def get_image_class(self,label):
        return self.data[np.array(self.targets)==label]


