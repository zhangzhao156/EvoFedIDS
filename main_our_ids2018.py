# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler,normalize
import joblib
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
import sklearn
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support,accuracy_score
from sklearn.model_selection import train_test_split
import copy
from collections import Counter, Iterable, defaultdict
from itertools import chain,combinations, permutations
import random
from scipy.spatial.distance import cdist, euclidean
import argparse
from Net import CNN
import time
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from thop import profile,clever_format
from fvcore.nn import FlopCountAnalysis, parameter_count_table,flop_count,parameter_count

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# convert a list of list to a list [[],[],[]]->[,,]
def flatten(items):
    """Yield items from any nested iterable; see Reference."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x



def preprocess(raw_data):
    raw_data.drop(
        ['Dst Port', 'Protocol', 'Timestamp', 'Flow Duration', 'Fwd URG Flags', 'Bwd URG Flags', 'CWE Flag Count',
         'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg'], axis=1, inplace=True)
    attacks_mapping = {'Benign':0,'Bot': 1,'DoS attacks-Hulk':2,'FTP-BruteForce':3,'SSH-Bruteforce':4,'DDOS attack-HOIC':6,
                       'DoS attacks-GoldenEye': 7, 'DoS attacks-Slowloris': 8, 'DDOS attack-LOIC-UDP': 9,
                       'Brute Force -XSS':10,'SQL Injection':11,
                       'Infilteration':12,'DoS attacks-SlowHTTPTest':13}
    raw_data.iloc[:, 68] = (raw_data.iloc[:, 68]).map(attacks_mapping)
    # raw_data_d = raw_data.iloc[0:160000]
    # return raw_data_d
    return raw_data

def preprocess_v2(raw_data):
    raw_data.drop(
        ['Flow ID', 'Src IP', 'Src Port', 'Dst IP','Dst Port', 'Protocol', 'Timestamp', 'Flow Duration', 'Fwd URG Flags', 'Bwd URG Flags', 'CWE Flag Count',
         'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg'], axis=1, inplace=True)
    attacks_mapping = {'DDoS attacks-LOIC-HTTP': 5}
    raw_data.iloc[:, 68] = (raw_data.iloc[:, 68]).map(attacks_mapping)
    return raw_data
    # raw_data_d = raw_data.iloc[0:160000]
    # return raw_data_d

def readdataset():
    raw_data0 = pd.read_csv('NewCleanedData/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv-Benign.csv', header=0)
    raw_data1 = pd.read_csv('NewCleanedData/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv-Bot.csv', header=0)
    raw_data2 = pd.read_csv('NewCleanedData/Friday-16-02-2018_TrafficForML_CICFlowMeter.csv-DoS attacks-Hulk.csv',
                            header=0)
    raw_data3 = pd.read_csv('NewCleanedData/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv-FTP-BruteForce.csv',
                            header=0)
    raw_data4 = pd.read_csv('NewCleanedData/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv-SSH-Bruteforce.csv',
                            header=0)
    raw_data5 = pd.read_csv(
        'NewCleanedData/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv-DDoS attacks-LOIC-HTTP.csv', header=0)
    raw_data6 = pd.read_csv(
        'NewCleanedData/Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv-DDOS attack-HOIC.csv', header=0)
    raw_data7 = pd.read_csv(
        'NewCleanedData/Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv-DoS attacks-GoldenEye.csv', header=0)
    raw_data8 = pd.read_csv(
        'NewCleanedData/Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv-DoS attacks-Slowloris.csv', header=0)
    raw_data9 = pd.read_csv(
        'NewCleanedData/Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv-DDOS attack-LOIC-UDP.csv', header=0)
    ## raw_data10 = pd.read_csv('NewCleanedData/Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv-Brute Force -Web.csv', header=0)
    # raw_data10 = pd.read_csv('NewCleanedData/Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv-Brute Force -XSS.csv', header=0)

    # raw_data11 = pd.read_csv('NewCleanedData/Friday-23-02-2018_TrafficForML_CICFlowMeter.csv-SQL Injection.csv', header=0)
    # raw_data12 = pd.read_csv('NewCleanedData/Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv-Infilteration.csv', header=0)
    # raw_data13 = pd.read_csv('NewCleanedData/Friday-16-02-2018_TrafficForML_CICFlowMeter.csv-DoS attacks-SlowHTTPTest.csv', header=0)

    raw_data0 = preprocess(raw_data0)
    raw_data1 = preprocess(raw_data1)
    raw_data2 = preprocess(raw_data2)
    raw_data3 = preprocess(raw_data3)
    raw_data4 = preprocess(raw_data4)
    raw_data5 = preprocess_v2(raw_data5)
    raw_data6 = preprocess(raw_data6)
    raw_data7 = preprocess(raw_data7)
    raw_data8 = preprocess(raw_data8)
    raw_data9 = preprocess(raw_data9)
    # raw_data10 = preprocess(raw_data10)
    # raw_data11 = preprocess(raw_data11)
    # raw_data12 = preprocess(raw_data12)
    # raw_data13 = preprocess(raw_data13)
    raw_data = pd.concat([raw_data0, raw_data1, raw_data2, raw_data3, raw_data4, raw_data5, raw_data6,
                          raw_data7, raw_data8, raw_data9]) ##, raw_data10, raw_data11, raw_data12, raw_data13
    feature_size = raw_data.shape[1] - 1
    # upsampling = RandomOverSampler(sampling_strategy={10:5000}, random_state=0) #,11:5000
    # Xt, yt = upsampling.fit_resample(raw_data.iloc[:, 0:feature_size], raw_data.iloc[:, feature_size])
    Xt, yt = raw_data.iloc[:, 0:feature_size], raw_data.iloc[:, feature_size]
    scaler = MinMaxScaler()
    scaler.fit(Xt)
    Xt_norm = scaler.transform(Xt)
    df = pd.DataFrame(Xt_norm, index=yt)
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=25)
    df_train = shuffle(df_train)
    np_features_train = df_train.values
    np_features_train = np_features_train[:, np.newaxis, :]
    np_label_train = df_train.index.values.ravel()
    print('train', sorted(Counter(np_label_train).items()))
    df_test = shuffle(df_test)
    features_test = df_test.values
    np_features_test = np.array(features_test)
    np_features_test = np_features_test[:, np.newaxis, :]
    np_label_test = df_test.index.values.ravel()
    print('test', sorted(Counter(np_label_test).items()))
    return np_features_train, np_label_train, np_features_test, np_label_test

# def readdataset(is_training1):
#     raw_data0 = pd.read_csv('NewCleanedData/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv-Benign.csv', header=0)
#     raw_data1 = pd.read_csv('NewCleanedData/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv-Bot.csv', header=0)
#     raw_data2 = pd.read_csv('NewCleanedData/Friday-16-02-2018_TrafficForML_CICFlowMeter.csv-DoS attacks-Hulk.csv',
#                             header=0)
#     raw_data3 = pd.read_csv('NewCleanedData/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv-FTP-BruteForce.csv',
#                             header=0)
#     raw_data4 = pd.read_csv('NewCleanedData/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv-SSH-Bruteforce.csv',
#                             header=0)
#     raw_data5 = pd.read_csv(
#         'NewCleanedData/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv-DDoS attacks-LOIC-HTTP.csv', header=0)
#     raw_data6 = pd.read_csv(
#         'NewCleanedData/Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv-DDOS attack-HOIC.csv', header=0)
#
#     raw_data0 = preprocess(raw_data0)
#     raw_data1 = preprocess(raw_data1)
#     raw_data2 = preprocess(raw_data2)
#     raw_data3 = preprocess(raw_data3)
#     raw_data4 = preprocess(raw_data4)
#     raw_data5 = preprocess_v2(raw_data5)
#     raw_data6 = preprocess(raw_data6)
#
#     if is_training1 == 'Training':
#         train0 = raw_data0.iloc[0:112000]
#         train1 = raw_data1.iloc[0:112000]
#         train2 = raw_data2.iloc[0:112000]
#         train3 = raw_data3.iloc[0:112000]
#         train4 = raw_data4.iloc[0:112000]
#         train5 = raw_data5.iloc[0:112000]
#         train6 = raw_data6.iloc[0:112000]
#         df_train = pd.concat([train0, train1, train2, train3, train4, train5, train6])  #
#         df_train = shuffle(df_train)
#         features_train = df_train.iloc[:, 0:df_train.shape[1] - 1]
#         np_features_train = np.array(features_train)
#         scaler = MinMaxScaler()
#         scaler.fit(np_features_train)
#         np_features_train = scaler.transform(np_features_train)
#         joblib.dump(scaler, "scaler.gz")
#         np_features = np_features_train[:, np.newaxis, :]
#         labels_train = df_train.iloc[:, df_train.shape[1] - 1:]
#         labels_train = labels_train.values.ravel()
#         np_labels = np.array(labels_train)
#     else:
#         test0 = raw_data0.iloc[112000:160000]
#         test1 = raw_data1.iloc[112000:160000]
#         test2 = raw_data2.iloc[112000:160000]
#         test3 = raw_data3.iloc[112000:160000]
#         test4 = raw_data4.iloc[112000:160000]
#         test5 = raw_data5.iloc[112000:160000]
#         test6 = raw_data6.iloc[112000:160000]
#
#         df_test = pd.concat([test0, test1, test2, test3, test4, test5, test6])  ##
#         df_test = shuffle(df_test)
#         features_test = df_test.iloc[:, 0:df_test.shape[1] - 1]
#         np_features_test = np.array(features_test)
#         scaler = joblib.load('scaler.gz')
#         np_features_test = scaler.transform(np_features_test)
#         np_features = np_features_test[:, np.newaxis, :]
#         labels_test = df_test.iloc[:, df_test.shape[1] - 1:]
#         labels_test = labels_test.values.ravel()
#         np_labels = np.array(labels_test)
#     return np_features, np_labels


class ReadData(Dataset):
    def __init__(self, x_tra, y_tra):
        self.x_train = x_tra
        self.y_train = y_tra

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, item):
        image, label = self.x_train[item], self.y_train[item]
        img = torch.from_numpy(image)
        label = torch.from_numpy(np.asarray(label))
        return img, label


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.features, self.labels = self.dataset[self.idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def data_from_labels(imgs, targets, class_set):
    class_mask = np.any((targets[:, None] == class_set[None, :]), axis=-1)
    return imgs[class_mask],targets[class_mask]

def xy_from_labels(x,y,class_set,n):
    # print('xy',x.shape,y.shape) ##[-1,1,70]
    for i,c in enumerate(class_set):
        class_mask = np.any((y[:, None] == c), axis=-1)
        # print('class_mask',class_mask)
        if i == 0:
            m_x = x[class_mask][:n,:]
            m_y = y[class_mask][:n,]
        else:
            m_x = np.concatenate((m_x,x[class_mask][:n,:]), axis=0)
            m_y = np.concatenate((m_y, y[class_mask][:n, ]), axis=0)
        # print(c,'m_y',m_y)
    return m_x,m_y



def dataset_from_labels(imgs, targets, class_set):
    class_mask = np.any((targets[:, None] == class_set[None, :]),axis=-1)### 只要该行有一个True,则返回True
    # print(class_mask)
    # print('class_mask',class_mask.shape)
    return ReadData(imgs[class_mask], targets[class_mask])

# def iid(dataset, num_users):
#     """
#     Sample I.I.D. client data from dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
#     num_items = int(len(dataset) / num_users)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items,
#                                              replace=False))  # Generates random samples from all_idexs,return a array with size of num_items
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users

def iid(dataset, client_list):
    """
    Sample I.I.D. client data from dataset
    :param dataset:
    :param client_list:
    :return: dict of image index
    """
    num_items = int(len(dataset) / len(client_list))
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in client_list:
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))  # Generates random samples from all_idexs,return a array with size of num_items
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def avgomega(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def avgproto(local_proto):
    proto_avg = local_proto[0]
    for i in range(1,len(local_proto)):
        proto_avg += local_proto[i]
    proto_avg = torch.div(proto_avg,len(local_proto))
    return proto_avg

def avgproto_dict(local_proto):
    proto_all = local_proto[0].copy()
    for i in range(1, len(local_proto)):
        proto_all.update(local_proto[i])
    proto_avg_dict = {}
    proto_avg = []
    # print('proto global classes',proto_all.keys())
    proto_all_keys = np.sort(np.array(list(proto_all.keys())))
    print('proto global classes', proto_all_keys)
    for k in proto_all_keys:
        proto_avg_dict[k] = torch.zeros([1,embedding_size]).to(device)
        num = 0
        for i in range(len(local_proto)):
            if k in local_proto[i].keys():
                proto_avg_dict[k] += local_proto[i][k]
                num += 1
        proto_avg_dict[k] = torch.div(proto_avg_dict[k],num)
        # print('div',proto_avg_dict[k].size())
        proto_avg.append(proto_avg_dict[k])
    # proto_avg=torch.from_numpy(np.array(proto_avg))
    proto_avg =torch.stack(proto_avg, dim=0).squeeze(1)
    # print('proto_avg',proto_avg.size())
    return proto_avg

def test_prototype(net_g, datatest, global_proto):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    total = 0
    data_pred = []
    data_label = []
    data_loader = DataLoader(datatest, batch_size=test_BatchSize, shuffle=True)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        data, target = Variable(data).to(device), Variable(target).type(torch.LongTensor).to(device)
        features = net_g(data)
        # print('global_proto',global_proto.is_cuda,'features',features.is_cuda)
        dist = torch.pow(global_proto[None, :] - features[:, None], 2).sum(dim=2)  # Squared euclidean distance
        preds = F.log_softmax(-dist, dim=1)  ### 数值上rescale，size=[qurysamples_num,classes_num]
        test_loss +=  F.cross_entropy(preds, target).item()
        correct += (preds.argmax(dim=1) == target).cpu().float().sum()
        total += target.size(0)
        data_pred.append(preds.argmax(dim=1).cpu().detach().data.tolist())
        data_label.append(target.cpu().detach().data.tolist())

    list_data_label = list(flatten(data_label))
    list_data_pred = list(flatten(data_pred))
    all_report = precision_recall_fscore_support(list_data_label, list_data_pred, average='weighted')
    all_precision = all_report[0]
    all_recall = all_report[1]
    all_fscore = all_report[2]
    print('all_precision',all_precision,'all_recall',all_recall,'all_fscore',all_fscore)
    print(classification_report(list_data_label, list_data_pred))
    print(confusion_matrix(list_data_label, list_data_pred))
    test_loss /= (idx+1)
    accuracy = 100. * correct.type(torch.FloatTensor) / (total)
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} {:.2f}'.format(
        test_loss, correct, total, accuracy))
    return accuracy, test_loss

def test_img(net_g, datatest):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_pred = []
    data_label = []
    x = datatest.x_train
    y = datatest.y_train
    anomaly_list = [i for i in range(len(y)) if y[i] != 0]
    y[anomaly_list] = 1
    dataset_test = ReadData(x,y)
    data_loader = DataLoader(dataset_test, batch_size=test_BatchSize)
    loss = torch.nn.CrossEntropyLoss()
    for idx, (data, target) in enumerate(data_loader):
        data, target = Variable(data).to(device), Variable(target).type(torch.LongTensor).to(device)
        # data, target = Variable(data), Variable(target).type(torch.LongTensor)
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += loss(log_probs, target).item()
        # test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.detach().max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.detach().view_as(y_pred)).long().cpu().sum()
        data_pred.append(y_pred.cpu().detach().data.tolist())
        data_label.append(target.cpu().detach().data.tolist())
    list_data_label = list(flatten(data_label))
    list_data_pred = list(flatten(data_pred))
    print(classification_report(list_data_label, list_data_pred))
    print(confusion_matrix(list_data_label, list_data_pred))
    print('test_loss', test_loss)
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss


def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)

def test(train_set, test_set, model, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    classes, _ = torch.unique(train_labels).sort()
    prototypes = []
    for c in classes:
        p = train_embeddings[torch.where(train_labels == c)[0]].mean(dim=0)  # Average class feature vectors
        prototypes.append(p)
    prototypes = torch.stack(prototypes, dim=0)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    test_labels = test_labels.squeeze(1)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, prototypes, test_labels, classes, False
    )
    print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))
    ### NCM classification
    dist = torch.pow(prototypes[None, :] - test_embeddings[:, None], 2).sum(dim=2)
    preds = F.log_softmax(-dist, dim=1)
    correct = (preds.argmax(dim=1) == test_labels).cpu().float().sum()
    NCM_accuracy = 100. * correct.type(torch.FloatTensor) / (test_labels.size(0))
    print('NCM_accuracy',NCM_accuracy)
    return prototypes

def test2(train_set, model):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    classes, _ = torch.unique(train_labels).sort()
    # prototypes = []
    prototypes = {}
    for c in classes:
        p = train_embeddings[torch.where(train_labels == c)[0]].mean(dim=0)  # Average class feature vectors
        # prototypes.append(p)
        prototypes[c.item()]=p
    # prototypes = torch.stack(prototypes, dim=0)
    return prototypes



# def update_memory_proto(train_set, model, m):
#     train_embeddings, train_labels = get_all_embeddings(train_set, model)
#     classes, _ = torch.unique(train_labels).sort()
#     prototypes = {}
#     memory = np.empty((0,1,68))
#     memeory_class_list = np.empty((0))
#
#     cls_number = [torch.where(train_labels == cls)[0].size(0) for cls in classes]
#     # print('#### the candidate counter for memory', cls_number)
#     all_men_number = m
#     all_cls_number = len(cls_number)
#     cls_memory_size = {}
#     for i in range(len(cls_number)):
#         if cls_number[i] < m/len(cls_number):
#             cls_memory_size[classes[i].item()]=cls_number[i]
#             all_men_number -= cls_number[i]
#             all_cls_number -= 1
#     for i in range(len(cls_number)):
#         if cls_number[i] >= m / len(cls_number):
#             cls_memory_size[classes[i].item()] = int(all_men_number/all_cls_number)
#     # print('cls_memory_size',cls_memory_size)
#
#     std_proto = {}
#     for c in classes:
#         class_embeddings = train_embeddings[torch.where(train_labels == c)[0]]
#         p = class_embeddings.mean(dim=0)  # Average class feature vectors
#         p_std = class_embeddings.std(dim=0).sum()
#         std_proto[c.item()] = p_std.item()
#         # print('class_embeddings',class_embeddings.size(),'p',p.size(),(class_embeddings-p).size())
#         dd = np.linalg.norm((class_embeddings-p).cpu(), axis=1)
#         # print(dd)
#         # print('c',c.cpu())
#         # memeory_class_list.append(c.item())
#         m = cls_memory_size[c.item()]
#         #### random selection
#         idx = np.random.permutation(torch.where(train_labels == c)[0].cpu())
#         #### prototype guided selection
#         # idx_sort =np.argsort(-dd)
#         # # print('before idx',(torch.where(train_labels == c)[0].cpu()))
#         # idx = (torch.where(train_labels == c)[0].cpu())[idx_sort]
#         # # print('sort idx2',idx2)
#         if m < dd.shape[0]:
#             # idx = np.argpartition(dd, m)
#             # print('22222',(train_set.x_train)[idx[:m - 1]].shape,(train_set.y_train)[idx[:m - 1]].shape)
#             memory = np.append(memory,(train_set.x_train)[idx[:m]],axis=0)
#             memeory_class_list = np.append( memeory_class_list,(train_set.y_train)[idx[:m]],axis=0)
#         else:
#             m = dd.shape[0]
#             # idx = np.argpartition(dd, m)
#             # print('22222', (train_set.x_train)[idx[:m - 1]].shape,(train_set.y_train)[idx[:m - 1]].shape)
#             memory = np.append(memory,(train_set.x_train)[idx[:m]],axis=0)
#             memeory_class_list = np.append( memeory_class_list,(train_set.y_train)[idx[:m]],axis=0)
#         prototypes[c.item()]=p
#         # prototypes.append(p)
#     # prototypes = torch.stack(prototypes, dim=0)
#     # print('33333',memeory_class_list.shape,memory.shape)
#     print('memeory_class_list',Counter(memeory_class_list))
#     print('std', std_proto)
#     return prototypes,memory,memeory_class_list

def update_memory_proto(train_set, model, m):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    classes, _ = torch.unique(train_labels).sort()
    prototypes = {}
    std_proto = []
    for c in classes:
        class_embeddings = train_embeddings[torch.where(train_labels == c)[0]]
        p = class_embeddings.mean(dim=0)  # Average class feature vectors
        prototypes[c.item()] = p
        p_std = class_embeddings.std(dim=0).sum()
        std_proto.append(p_std.item())
    memory = np.empty((0,1,68))
    memeory_class_list = np.empty((0))
    std_proto_norm = std_proto/np.array(std_proto).sum()
    print('std_proto',std_proto)
    print('std_proto_norm', std_proto_norm)
    cls_number = [torch.where(train_labels == cls)[0].size(0) for cls in classes]
    print('#### the candidate counter for memory', cls_number)
    # cls_number_weight = np.array(cls_number)/train_labels.size(0)
    cls_memory_size = {}
    all_men_number = m
    for i in range(len(cls_number)):
        if cls_number[i] < m*std_proto_norm[i]:
            cls_memory_size[classes[i].item()]=cls_number[i]
            all_men_number -= cls_number[i]
            std_proto_norm[i] = 0.0
    std_proto_norm = std_proto_norm/std_proto_norm.sum()

    for i in range(len(cls_number)):
        if std_proto_norm[i] != 0.0:
            cls_memory_size[classes[i].item()] = int(std_proto_norm[i]*all_men_number)
        # else:
        #     cls_memory_sizemory_size[classes[i].item()] = int(std_proto_norm[i]*m)
    # std_proto_norm = std_proto_norm/std_proto_norm.sum()
    # for i in range(len(cls_number)):
    #     cls_memory_size[classes[i].item()] = int(m*std_proto_norm[i])
    # print('cls_memory_size',cls_memory_size)

    k = 0
    for c in classes:
        cls_m = cls_memory_size[c.item()]
        #### random selection
        idx = np.random.permutation(torch.where(train_labels == c)[0].cpu())
        if cls_m < cls_number[k]:
            # idx = np.argpartition(dd, m)
            # print('22222',(train_set.x_train)[idx[:m - 1]].shape,(train_set.y_train)[idx[:m - 1]].shape)
            memory = np.append(memory,(train_set.x_train)[idx[:cls_m]],axis=0)
            memeory_class_list = np.append( memeory_class_list,(train_set.y_train)[idx[:cls_m]],axis=0)
        else:
            cls_m = cls_number[k]
            # idx = np.argpartition(dd, m)
            # print('22222', (train_set.x_train)[idx[:m - 1]].shape,(train_set.y_train)[idx[:m - 1]].shape)
            memory = np.append(memory,(train_set.x_train)[idx[:cls_m]],axis=0)
            memeory_class_list = np.append( memeory_class_list,(train_set.y_train)[idx[:cls_m]],axis=0)
        k += 1

    # prototypes = torch.stack(prototypes, dim=0)
    # print('33333',memeory_class_list.shape,memory.shape)
    print('memeory_class_list',Counter(memeory_class_list))
    return prototypes,memory,memeory_class_list

def consolidate(Model, Weight, MEAN_pre, epsilon):
    OMEGA_current = {n: p.data.clone().zero_() for n, p in Model.named_parameters()}
    for n, p in Model.named_parameters():
        p_current = p.detach().clone()
        p_change = p_current - MEAN_pre[n]
        OMEGA_add = torch.max(Weight[n], Weight[n].clone().zero_()) / (p_change ** 2 + epsilon)
        OMEGA_current[n] = OMEGA_add
    # MEAN_current = {n: p.data for n, p in Model.named_parameters()}
    return OMEGA_current#,MEAN_current



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Nmemory', type=int, nargs='?', default=2000, help="memory size")
    parser.add_argument('--lamda', type=float, nargs='?', default=0.01, help="memory size")
    parser.add_argument('--baseclass', type=int, nargs='?', default=3, help="base_class_number")
    parser.add_argument('--increclass', type=int, nargs='?', default=1, help="incre_class_number")
    parser.add_argument('--E', type=int, nargs='?', default=3, help="E")
    parser.add_argument('--T', type=int, nargs='?', default=1, help="T")#24
    parser.add_argument('--pertaskT', type=int, nargs='?', default=1, help="per_task_T")#3
    parser.add_argument('--Nclient', type=int, nargs='?', default=1, help="N")#10

    args = parser.parse_args()
    E = args.E
    base_class_number = args.baseclass ##remaining 10-3=7 classes for incremental learning
    incre_class_number = args.increclass #
    T = args.T #8
    per_task_T = args.pertaskT #1
    epsilon = 0.0001
    lamda = args.lamda#0.01
    print('lamda',lamda)
    # num_clients = 10 #5
    # part_num_clients = int(num_clients*0.5)

    num_clients = args.Nclient  # 5
    part_num_clients = int(num_clients)

    part_clients_lists = list(combinations([i for i in range(0, num_clients)], part_num_clients)) ##[(0,1),(1,2),(0,2)]
    random.seed(1)
    random.shuffle(part_clients_lists)
    batch_size = 512
    test_BatchSize = 512
    # N_Classes = 6
    embedding_size = 8 #5
    N_memory = args.Nmemory #1000
    print('N_memory',N_memory)

    net_global = CNN(embedding_size).double().to(device)
    w_glob = net_global.state_dict()
    #####
    # crit = torch.nn.CrossEntropyLoss()
    ##### triplet loss
    # distance = distances.CosineSimilarity()
    # reducer = reducers.ThresholdReducer(low=0)
    # loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
    # mining_func = miners.TripletMarginMiner(
    #     margin=0.2, distance=distance, type_of_triplets="semihard")
    #### contrastive loss
    # distance = distances.CosineSimilarity()
    # reducer = reducers.ThresholdReducer(low=0)
    # loss_func = losses.ContrastiveLoss(pos_margin=1, neg_margin=0, distance=distance, reducer=reducer)
    #### supervised contrastive loss
    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    mining_func = miners.TripletMarginMiner(
        margin=0.2, distance=distance, type_of_triplets="semihard")
    loss_func = losses.SupConLoss(temperature=0.1, distance=distance, reducer=reducer)
    ##### centroid triplet loss
    # loss_func = losses.CentroidTripletLoss()
    #######
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

    net_global.train()

    x_train, y_train, x_test, y_test = readdataset()
    # print('data', x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # omega_current, mean_current = {}, {}
    memory = {}
    memory_classlabel = {}
    for i in range(num_clients):
        # omega_current[i] = {}
        # mean_current[i] = {}
        memory[i] = np.empty((0,1,68))
        memory_classlabel[i] = []

    for interation in range(T):
        w_locals, loss_locals = [], []
        proto_locals = []
        omega_locals = []
        task_id = int(interation/per_task_T)

        if (interation % per_task_T == 0):
            client_list = [i for i in part_clients_lists[int((interation / per_task_T) % len(part_clients_lists))]]

        if task_id == 0:
            # client_list = [i for i in range(num_clients)]
            class_set_stage_train = np.array([i for i in range(base_class_number)])
            class_set_stage_test = np.array([i for i in range(base_class_number)])
        else:
            # if (interation%per_task_T==0):
            #     # client_list = [i for i in range(num_clients-1)]
            #     # client_list = random.sample([i for i in range(num_clients)], int(num_clients * 0.9))
            #     client_list = [i for i in part_clients_lists[int((interation/per_task_T)%len(part_clients_lists))]]
            #     # print('client_list',client_list)

            class_set_stage_train = np.array([i for i in range(base_class_number+(task_id-1)*incre_class_number,base_class_number+task_id*incre_class_number)])
            class_set_stage_test = np.array([i for i in range(base_class_number+task_id*incre_class_number)])
            class_set_stage_old = list(set(class_set_stage_test.tolist())-set(class_set_stage_train.tolist()))

        dataset_train_stage = dataset_from_labels(imgs=x_train, targets=y_train, class_set=class_set_stage_train)
        dataset_test_stage = dataset_from_labels(imgs=x_test, targets=y_test, class_set=class_set_stage_test)
        dict_clients_stage = iid(dataset_train_stage, client_list)

        print(' ')
        print('********** current classes',class_set_stage_train,class_set_stage_test)
        for client in client_list:
            mean_current = copy.deepcopy(net_global).to(device)
            net = copy.deepcopy(net_global).to(device)
            net.train()
            opt_net = torch.optim.Adam(net.parameters(), lr=0.01)
            # opt_net = torch.optim.Adam(net.parameters(), lr=0.001)
            # opt_net = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.7) ##, momentum=0.5

            # print('## interation', interation, 'client', client)
            idx_traindataset = DatasetSplit(dataset_train_stage, dict_clients_stage[client])

            x_train_stage, y_train_stage = idx_traindataset.features, idx_traindataset.labels

            if task_id > 0:
                # print('x_train_stage',x_train_stage.shape,'y_train_stage',y_train_stage.shape)
                x_train_stage = np.concatenate((x_train_stage, memory[client]), axis=0)
                y_train_stage = np.concatenate((y_train_stage, memory_classlabel[client]), axis=0)
                # print('add x_train_stage',x_train_stage.shape,'y_train_stage',y_train_stage.shape)
                # print('add mem counter',Counter(y_train_stage))
                p = np.random.permutation(y_train_stage.shape[0])
                idx_traindataset = ReadData(x_train_stage[p],y_train_stage[p])
                ## 数据集中各类样本数量
                # yy = idx_traindataset.y_train
                # print('0', torch.where(yy == torch.tensor(0))[0].size(0), '1', torch.where(yy == 1)[0].size(0), '2',
                #       torch.where(yy == 2)[0].size(0),'3', torch.where(yy == 3)[0].size(0),'4', torch.where(yy == 4)[0].size(0))
            else:
                idx_traindataset = ReadData(x_train_stage.numpy(), y_train_stage.numpy())

            ldr_train = DataLoader(idx_traindataset, batch_size=256, shuffle=True)
            dataset_size = len(ldr_train.dataset)
            epochs_per_task = E
            t0 = time.time()

            mean_pre = {n: p.clone().detach() for n, p in net.named_parameters()}
            W = {n: p.clone().detach().zero_() for n, p in net.named_parameters()}

            for epoch in range(1, epochs_per_task + 1):
                correct = 0
                total = 0
                train_loss = 0
                data_pred = []
                data_label = []
                # old_prototypes = torch.empty((class_set_stage1.shape[0],5))
                for batch_idx, (images, labels) in enumerate(ldr_train):
                    old_par = {n: p.clone().detach() for n, p in net.named_parameters()}

                    images,labels = Variable(images).to(device),Variable(labels).type(torch.LongTensor).to(device)
                    # print('batch',batch_idx,images.size(),labels)

                    # macs, params = profile(net, inputs=(images,))  # ,verbose=False
                    # print("MACs", macs)
                    # print("p", params)
                    # macs, params = clever_format([macs, params], "%.3f")
                    # print(macs, params)
                    # print("@@@@@@@@@@@@@@")
                    # print(parameter_count_table(net))
                    # flops = FlopCountAnalysis(net, inputs=(images,))
                    # print(flops.by_module_and_operator())


                    net.zero_grad()
                    features = net(images)
                    #### triplet loss / supcon loss
                    # indices_tuple = mining_func(features, labels)
                    # ce_loss = loss_func(features, labels, indices_tuple)
                    #### constractive loss
                    ce_loss = loss_func(features, labels)
                    #### centroidtriplet loss
                    # ce_loss = loss_func(features, labels)

                    grad_params = torch.autograd.grad(ce_loss, net.parameters(), create_graph=True) ##, retain_graph=True
                    if task_id == 0:
                        si_loss = torch.DoubleTensor([0.0]).to(device)
                    else:
                        sum_si_loss = 0.0
                        for n, p in net.named_parameters():
                            # omega = Variable(omega_current[client][n])
                            # current = Variable(mean_current[client][n])
                            omega = Variable(omega_glob[n])
                            current = Variable(w_glob[n])
                            sum_si_loss += (omega * (p - current) ** 2).sum()
                        si_loss = lamda * sum_si_loss
                        # print('si_loss',si_loss)

                    loss = ce_loss + si_loss

                    # si_loss = torch.DoubleTensor([0.0]).to(device)
                    # loss = ce_loss

                    loss.backward()
                    opt_net.step()
                    train_loss += loss.item()

                    j = 0
                    for n, p in net.named_parameters():
                        W[n] -= (grad_params[j].clone().detach()) * (p.detach() - old_par[n])
                        j += 1

                # print('Train Epoch:{}\tCE Loss:{:.4f}\tSI Loss:{:.4f}\tLoss:{:.4f}'.format(epoch, ce_loss.item(), si_loss.item(), train_loss/(batch_idx+1)))


            w_locals.append(copy.deepcopy(net.state_dict()))
            t1 = time.time()
            print('client:\t', client, 'trainingtime:\t', str(t1 - t0))

            if (interation+1)%per_task_T != 0:
                old_prototypes = test2(idx_traindataset, net)
            else:
                ### update the memory after every task
                old_prototypes, memory[client],memory_classlabel[client]= update_memory_proto(train_set=idx_traindataset, model=net, m=N_memory)
            # print('old_prototypes',old_prototypes)
            # for k in old_prototypes.keys():
            #     print(k,old_prototypes[k].tolist())

            proto_locals.append(old_prototypes)

            # omega_current[client], mean_current[client] = consolidate(Model=net, Weight=W, MEAN_pre=mean_pre, epsilon=epsilon)
            omega_current = consolidate(Model=net, Weight=W, MEAN_pre=mean_pre, epsilon=epsilon)
            omega_locals.append(omega_current)



        w_glob = FedAvg(w_locals)
        proto_glob = avgproto_dict(proto_locals)
        omega_glob = avgomega(omega_locals)
        # proto_glob = avgproto(proto_locals)
        net_global.load_state_dict(w_glob)
        net_global.eval()
        # print('global_protos_stage', proto_glob)
        acc_test, loss_test = test_prototype(net_global, dataset_test_stage, proto_glob)
        print("Testing accuracy: {:.2f}".format(acc_test))


