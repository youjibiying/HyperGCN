# parse arguments ([ConfigArgParse](https://github.com/bw2/ConfigArgParse))
from config import config
import os, torch, numpy as np
from data import data
from model import model
import csv
def data_write_csv(filename,data):
    with open(filename, "a", encoding="utf-8", newline='') as w:
        writer = csv.writer(w)
        writer.writerow(data)


args = config.parse()



# seed

torch.manual_seed(args.seed)
np.random.seed(args.seed)



# gpu, seed
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"        
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ['PYTHONHASHSEED'] = str(args.seed)



# load data

if __name__ == '__main__':
    splits = [i+1 for i in range(10)]
    results = []
    for split in splits:
        print(f"split: {split}/{splits}")
        args.split = split
        dataset, train, test = data.load(args)
        #print("length of train is", len(train))



        # # initialise HyperGCN

        HyperGCN = model.initialise(dataset, args)



        # train and test HyperGCN
        HyperGCN,acc = model.train(HyperGCN, dataset, train, args,test)
        # acc=model.test(HyperGCN, dataset, test, args)

        results.append(acc.cpu().item())
        print('beat_acc:',acc.cpu().item())

    results = np.array(results)
    print(f"dataset={args.data}", #_{dataset}\n"
          f"avg_test_acc={results.mean()} \n"
          f"std={results.std()}")

    data_write_csv('records.csv',[args.data,args.dataset,args.depth,float(100*results.mean()), float(100*(1-results.mean()))])