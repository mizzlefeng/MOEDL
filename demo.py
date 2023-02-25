'''
@author: feng
'''
import pandas as pd
import random
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from Model import TMC
from torch import optim
from torch.autograd import Variable
import time
from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score,cohen_kappa_score,average_precision_score
from sklearn.preprocessing import label_binarize
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
# 参数设置
localtime=time.strftime("%m月%d日%H时%M分", time.localtime())

# 神经网络参数
batch_size = 256
epochs = 1000
lr = 0.01
lambda_epochs = 1
wd=0 # weight_decay
dropout=0.5

# 数据类型选择
data_type = 'demo'
# classes = 3
type_num = 4
# 数据路径
path = r'demodata/'

omic_data_total=["G","mi","M","P"]
omic_data1="G"
omic_data2="mi"
omic_data3="M"
omic_data4="P"

# 定义一个可以设置随机种子的函数
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.benchmark = False
     torch.backends.cudnn.deterministic = True

def show_parser():
    print("MOEDL start...........")
    print("batch-size:{}".format(batch_size))
    print("epochs:{}".format(epochs))
    print("lambda-epochs:{}".format(lambda_epochs))
    print("学习率:{}".format(lr))
    print("组学数据路径:{}".format(path))

def data_read_pre(path):
    data1 = pd.read_csv(os.path.join(path, data_type+"_{}.csv".format(omic_data1)),header=None)
    data2 = pd.read_csv(os.path.join(path, data_type+"_{}.csv".format(omic_data2)),header=None)
    data3 = pd.read_csv(os.path.join(path, data_type+"_{}.csv".format(omic_data3)),header=None)
    data4 = pd.read_csv(os.path.join(path, data_type+"_{}.csv".format(omic_data4)),header=None)
    label = pd.read_csv(os.path.join(path, data_type+"_label.csv"),header=None)

    #转换成2维数组的格式
    classes=len(set(label[0].tolist()))
    Y=np.array(label[0].tolist())
    
    data1 = data1.values  #从dataframe类型转换成Array of float64的形式
    data2 = data2.values
    data3 = data3.values
    data4 = data4.values
    return data1,data2,data3,data4,Y,classes

def Evaluation(y_true,y_pred,y_pred_prob):
    eval_indicator=dict()
    
    precision_macro=precision_score(y_true, y_pred, average="macro")
    recall_macro=recall_score(y_true, y_pred, average="macro")
    f1_macro=f1_score(y_true, y_pred, average="macro")
    f1_weighted=f1_score(y_true, y_pred, average="weighted")
    prob_sum=torch.sum(y_pred_prob, dim=1, keepdim=True)
    prob_sum=prob_sum+classes
    prob=y_pred_prob/prob_sum
    y_true_bin=label_binarize(y_true, classes=np.arange(classes))
    auc_macro=roc_auc_score(y_true_bin,prob,average='macro')
    auc_weighted=roc_auc_score(y_true_bin,prob,average='weighted')
    kappa = cohen_kappa_score(y_true,y_pred)
    aupr=average_precision_score(y_true_bin, prob)
    
    eval_indicator['kappa']=kappa
    eval_indicator['aupr']=aupr
    eval_indicator['precision_macro']=precision_macro
    eval_indicator['recall_macro']=recall_macro
    eval_indicator['f1_macro']=f1_macro
    eval_indicator['f1_weighted']=f1_weighted
    eval_indicator['auc_macro']=auc_macro
    eval_indicator['auc_weighted']=auc_weighted
    return eval_indicator

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(epoch):
    model.train()
    train_loss_meter = AverageMeter()
    correct_num, data_num = 0, 0
    predicted_all=torch.IntTensor([]).cuda()
    target_all=torch.IntTensor([]).cuda()
    evidence_all=torch.FloatTensor([]).cuda()
    
    for i, (dataE1, dataE2, dataE3, dataE4, target) in enumerate(trainLoader):
        data = dict()
        data[0] = dataE1
        data[1] = dataE2
        data[2] = dataE3
        data[3] = dataE4

        for v_num in range(len(data)):
            data[v_num] = Variable(data[v_num].cuda())

        target = target.long()
        target = Variable(target.cuda())

        optimizer.zero_grad()
        evidences, evidence_a, loss = model(data, target, epoch)
        loss.backward()
        optimizer.step()
        train_loss_meter.update(loss.item())
        
        with torch.no_grad():
            data_num += target.size(0)
            _, predicted = torch.max(evidence_a.data, 1)
            predicted_all=torch.cat((predicted_all,predicted)) 
            target_all=torch.cat((target_all,target)) 
            evidence_all=torch.cat((evidence_all,evidence_a),dim=0)
            correct_num += (predicted == target).sum().item()
    
    eval_indicator=Evaluation(target_all.cpu(),predicted_all.cpu(),evidence_all.cpu())
    if epoch % 20 == 0:
        print("~~~~>Train acc: {:.4f} | f1:{:.4f} | epoch:{}".format(correct_num/data_num,eval_indicator['f1_macro'],epoch))
    return train_loss_meter.avg,correct_num / data_num,eval_indicator

def test(epoch):
    model.eval()
    loss_meter = AverageMeter()
    correct_num, data_num = 0, 0
    predicted_all=torch.IntTensor([]).cuda()
    target_all=torch.IntTensor([]).cuda()
    evidence_all=torch.FloatTensor([]).cuda()
    for i, (dataE1, dataE2, dataE3, dataE4, target) in enumerate(testLoader):
        data = dict()
        data[0] = dataE1
        data[1] = dataE2
        data[2] = dataE3
        data[3] = dataE4

        for v_num in range(len(data)):
            data[v_num] = Variable(data[v_num].cuda())
        data_num += target.size(0)
        with torch.no_grad():
            target = Variable(target.long().cuda())
            evidences, evidence_a, loss = model(data, target, epoch)
            max_value, predicted = torch.max(evidence_a.data, 1)
            predicted_all=torch.cat((predicted_all,predicted)) 
            target_all=torch.cat((target_all,target)) 
            evidence_all=torch.cat((evidence_all,evidence_a),dim=0)
            correct_num += (predicted == target).sum().item()
            loss_meter.update(loss.item())
    eval_indicator=Evaluation(target_all.cpu(),predicted_all.cpu(),evidence_all.cpu())
    if epoch % 100 == 0:
        print("====>TEST acc: {:.4f} | f1:{:.4f}".format(correct_num/data_num,eval_indicator['f1_macro']))
    return loss_meter.avg, correct_num / data_num,eval_indicator

if __name__ == "__main__":
    total_stime = time.time()
    data1,data2,data3,data4,Y,classes = data_read_pre(path)
    skf = StratifiedKFold(n_splits=10,shuffle=True)
    dims = [[data1.shape[1]], [data2.shape[1]], [data3.shape[1]], [data4.shape[1]]]

    setup_seed(500)
    show_parser()
    k = 0
    
    costtr_all = []
    acctr_all = []

    costte_all = []      
    accte_all = []
    f1_weighted_te_all=[]
    f1_macro_te_all=[]
    auc_weighted_te_all=[]
    auc_macro_te_all=[]
    
    kappa_all = []
    precision_all=[]
    recall_all=[]
    aupr_all=[]

    start_time = time.time()
    for repeat in range(10):
        for train_index, test_index in skf.split(data1, Y.astype('int')):
            k+=1
            X_trainE1 = data1[train_index, :]
            X_trainE2 = data2[train_index, :]
            X_trainE3 = data3[train_index, :]
            X_trainE4 = data4[train_index, :]
 
            X_testE1 = data1[test_index, :]
            X_testE2 = data2[test_index, :]
            X_testE3 = data3[test_index, :]
            X_testE4 = data4[test_index, :]

            y_trainE = Y[train_index]
            y_testE = Y[test_index]

            print("第{}次循环·················".format(k))
            # 解决过拟合
            class_sample_count = np.array([len(np.where(y_trainE == t)[0]) for t in np.unique(y_trainE)])
            weight = 1 / class_sample_count
            samples_weight = np.array([weight[t] for t in y_trainE])
            samples_weight = torch.from_numpy(samples_weight)
            sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)

            # 封装数据
            trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_trainE1),
                                                          torch.FloatTensor(X_trainE2),
                                                          torch.FloatTensor(X_trainE3),
                                                           torch.FloatTensor(X_trainE4),
                                                          torch.FloatTensor(y_trainE.astype(int)))
            trainLoader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=batch_size, shuffle=False, num_workers=0,sampler=sampler)
            # print("train_loader的长度为：{}".format(len(trainLoader)))

            testDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_testE1),
                                                         torch.FloatTensor(X_testE2),
                                                         torch.FloatTensor(X_testE3),
                                                          torch.FloatTensor(X_testE4),
                                                         torch.FloatTensor(y_testE.astype(int)))
            testLoader = torch.utils.data.DataLoader(dataset=testDataset, batch_size=batch_size, shuffle=False, num_workers=0)
            # print("test_loader的长度为：{}".format(len(testLoader)))
          
            model = TMC(classes, type_num, dims, lambda_epochs,dropout)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            model.cuda()

            costte=[]
            accte,f1_weighted_te,f1_macro_te,auc_weighted_te,auc_macro_te,kappa_te,precision_te,recall_te,aupr_te= [],[],[],[],[],[],[],[],[]

            costtr=[]
            acctr=[]
            for epoch in range(1, epochs + 1):
                train_loss, train_acc,tr_indicators = train(epoch)
                costtr.append(train_loss)
                acctr.append(train_acc)

                test_loss, te_acc,te_indicators = test(epoch)
                costte.append(test_loss)
                accte.append(te_acc)
                f1_weighted_te.append(te_indicators['f1_weighted'])
                f1_macro_te.append(te_indicators['f1_macro'])
                auc_weighted_te.append(te_indicators['auc_weighted'])
                auc_macro_te.append(te_indicators['auc_macro'])
                kappa_te.append(te_indicators['kappa'])
                precision_te.append(te_indicators['precision_macro'])
                recall_te.append(te_indicators['recall_macro'])
                aupr_te.append(te_indicators['aupr'])
                
            # torch.save(model.state_dict(), 'model/Pre/{}_{}.pkl'.format(k,epoch))
            
            costtr_all.append(costtr)
            acctr_all.append(acctr)

            costte_all.append(costte)
            accte_all.append(accte)
            f1_weighted_te_all.append(f1_weighted_te)
            f1_macro_te_all.append(f1_macro_te)
            auc_weighted_te_all.append(auc_weighted_te)
            auc_macro_te_all.append(auc_macro_te)
            kappa_all.append(kappa_te)
            precision_all.append(precision_te)
            recall_all.append(recall_te)
            aupr_all.append(aupr_te)

    print("{}次10折交叉验证已经执行完...".format(10))
    costtr_all = np.array(costtr_all)
    acctr_all = np.array(acctr_all)
    costtr_mean = sum(costtr_all) / costtr_all.shape[0]
    costtr_std=np.std(costtr_all,axis=0)
    acctr_mean = sum(acctr_all) / acctr_all.shape[0]
    acctr_std=np.std(acctr_all,axis=0)
    
    # 验证的结果 cost、acc、precision、recall、f1、auc
    costte_all = np.array(costte_all)
    costte_mean = sum(costte_all) / costte_all.shape[0]
    costte_std=np.std(costte_all,axis=0)
    accte_all = np.array(accte_all)
    f1_weighted_te_all=np.array(f1_weighted_te_all)
    f1_macro_te_all=np.array(f1_macro_te_all)
    auc_weighted_te_all=np.array(auc_weighted_te_all)
    auc_macro_te_all=np.array(auc_macro_te_all)
    kappa_all=np.array(kappa_all)
    precision_all=np.array(precision_all)
    recall_all=np.array(recall_all)
    aupr_all=np.array(aupr_all)
    accte_mean = sum(accte_all) / accte_all.shape[0]
    accte_std=np.std(accte_all,axis=0)
    
    f1_weighted_te_mean=sum(f1_weighted_te_all) / f1_weighted_te_all.shape[0]
    f1_weighted_te_std=np.std(f1_weighted_te_all,axis=0)
    
    f1_macro_te_mean=sum(f1_macro_te_all) / f1_macro_te_all.shape[0]
    f1_macro_te_std=np.std(f1_macro_te_all,axis=0)
    
    auc_weighted_te_mean=sum(auc_weighted_te_all) / auc_weighted_te_all.shape[0]
    auc_weighted_te_std=np.std(auc_weighted_te_all,axis=0)
    
    auc_macro_te_mean=sum(auc_macro_te_all) / auc_macro_te_all.shape[0]
    auc_macro_te_std=np.std(auc_macro_te_all,axis=0)
    
    kappa_mean=sum(kappa_all) / kappa_all.shape[0]
    kappa_std=np.std(kappa_all,axis=0)
    
    precision_mean=sum(precision_all) / precision_all.shape[0]
    precision_std=np.std(precision_all,axis=0)
    
    recall_mean=sum(recall_all) / recall_all.shape[0]
    recall_std=np.std(recall_all,axis=0)
    
    aupr_mean=sum(aupr_all) / aupr_all.shape[0]
    aupr_std=np.std(aupr_all,axis=0)
    print('accmax:', round(max(accte_mean),4))
    print('f1_weighted:', round(max(f1_weighted_te_mean),4))
    print('f1_macro:', round(max(f1_macro_te_mean),4))
    print('auc_weighted:', round(max(auc_weighted_te_mean),4))
    print('auc_macro:', round(max(auc_macro_te_mean),4))
    print('kappa:', round(max(kappa_mean),4))
    print('precision:', round(max(precision_mean),4))
    print('recall:', round(max(recall_mean),4))
    print('aupr:', round(max(aupr_mean),4))

    accte_std_index=np.where(accte_mean==max(accte_mean))[0]
    accte_std=np.min(accte_std[accte_std_index])
    
    f1_weighted_te_std_index=np.where(f1_weighted_te_mean==max(f1_weighted_te_mean))[0]
    f1_weighted_te_std=np.min(f1_weighted_te_std[f1_weighted_te_std_index])
    
    f1_macro_te_std_index=np.where(f1_macro_te_mean==max(f1_macro_te_mean))[0]
    f1_macro_te_std=np.min(f1_macro_te_std[f1_macro_te_std_index])
    
    auc_weighted_te_std_index=np.where(auc_weighted_te_mean==max(auc_weighted_te_mean))[0]
    auc_weighted_te_std=np.min(auc_weighted_te_std[auc_weighted_te_std_index])
    
    auc_macro_te_std_index=np.where(auc_macro_te_mean==max(auc_macro_te_mean))[0]
    auc_macro_te_std=np.min(auc_macro_te_std[auc_macro_te_std_index])
    
    kappa_std_index=np.where(kappa_mean==max(kappa_mean))[0]
    kappa_std=np.min(kappa_std[kappa_std_index])
    
    precision_std_index=np.where(precision_mean==max(precision_mean))[0]
    precision_std=np.min(precision_std[precision_std_index])
    
    recall_std_index=np.where(recall_mean==max(recall_mean))[0]
    recall_std=np.min(recall_std[recall_std_index])
    
    aupr_std_index=np.where(aupr_mean==max(aupr_mean))[0]
    aupr_std=np.min(aupr_std[aupr_std_index])

    print("="*40)
    print("cancer type:{}".format(data_type))
    print("batch-size:{}".format(batch_size))
    print("epochs:{}".format(epochs))
    print("lambda-epochs:{}".format(lambda_epochs))
    print("学习率:{}".format(lr))

    end_time=time.time()
    runtime=time.strftime("%H:%M:%S", time.gmtime(end_time-start_time))  # 将运行时间转换成时分秒格式
    print("耗时：{}".format(runtime))

    with open("MOEDL 10fold.txt",'a')as f:
        f.write(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())+"\n")
        f.write("Acc: ({:.4f},{:.4f})".format(max(accte_mean),accte_std)+" ")
        f.write("F1_weighted: ({:.4f},{:.4f})".format(max(f1_weighted_te_mean),f1_weighted_te_std)+" ")
        f.write("F1_macro: ({:.4f},{:.4f})".format(max(f1_macro_te_mean),f1_macro_te_std)+" ")
        f.write("auc_weighted: ({:.4f},{:.4f})".format(max(auc_weighted_te_mean),auc_weighted_te_std)+" ")
        f.write("auc_macro: ({:.4f},{:.4f})".format(max(auc_macro_te_mean),auc_macro_te_std)+" ")
        f.write("kappa: ({:.4f},{:.4f})".format(max(kappa_mean),kappa_std)+" ")
        f.write("precision: ({:.4f},{:.4f})".format(max(precision_mean),precision_std)+" ")
        f.write("recall: ({:.4f},{:.4f})".format(max(recall_mean),recall_std)+" ")
        f.write("aupr: ({:.4f},{:.4f})".format(max(aupr_mean),aupr_std)+" ")
        f.write("\n" + "\n")  

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False

    plt.plot(np.squeeze(costtr_mean), '-r', np.squeeze(costte_mean), '-b')
    plt.ylabel('Total cost')
    plt.xlabel('epoch')
    title = 'Cost({}class)'.format(classes)
    plt.suptitle(title)
    plt.savefig(title + '.png', dpi=150)
    plt.close()

    plt.plot(np.squeeze(acctr_mean), '-r', np.squeeze(accte_mean), '-b')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    title ='Accuracy({}class)'.format(classes)
    plt.suptitle(title)
    plt.savefig(title + '.png', dpi=150)
    plt.close()

total_etime = time.time()
totalruntime = time.strftime("%H:%M:%S", time.gmtime(total_etime - total_stime))  # 将运行时间转换成时分秒格式
print("总耗时：{}".format(totalruntime))