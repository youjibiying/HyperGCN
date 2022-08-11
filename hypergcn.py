# parse arguments ([ConfigArgParse](https://github.com/bw2/ConfigArgParse))
from config import config
from data import data
from model import model
import csv,os
from datetime import datetime

def data_write_csv(filename,data):
    with open(filename, "a", encoding="utf-8", newline='') as w:
        writer = csv.writer(w)
        writer.writerow(data)


args = config.parse()


# gpu, seed
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
# os.environ["CUDA_VISIBLE_DEVItmuxCES"] = str(args.gpu)
os.environ['PYTHONHASHSEED'] = str(args.seed)

import torch, numpy as np

# seed

torch.manual_seed(args.seed)
np.random.seed(args.seed)





# load data

if __name__ == '__main__':
    elapsed_times=[]

    print(args)
    splits =[1+i for i in range(10)]
    results = []
    for split in splits:
        print(f"split: {split}/{splits}")
        args.split = split
        dataset, train, test = data.load(args)
        #print("length of train is", len(train))



        # # initialise HyperGCN

        HyperGCN = model.initialise(dataset, args)

        print(HyperGCN)

        # train and test HyperGCN
        HyperGCN,acc = model.train(HyperGCN, dataset, train, args,test)
        # acc=model.test(HyperGCN, dataset, test, args)

        results.append(acc.cpu().item())
        elapsed_times.append(time_elapsed)

        print('beat_acc:',acc.cpu().item())
        print(results)

    results = np.array(results)
    print(f"dataset={args.data}/{args.dataset}", #_{dataset}\n"
          f"avg_test_acc={results.mean()} \n"
          f"std={results.std()}")

    data_write_csv('records.csv',[f'{datetime.now()},{args.data}',args.dataset,args.depth,f"{float(100*results.mean()):.5}+-{results.std():.5}", float(100*(1-results.mean()))])