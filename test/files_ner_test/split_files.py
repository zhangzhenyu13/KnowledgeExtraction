import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--data", type=str,help="file to split into train, dev and test")
args=parser.parse_args()
train_file="train.json"
dev_file="dev.json"
test_file="test.json"
with open(args.data, "r", encoding="utf-8") as f:
    with open(train_file, "w", encoding="utf-8") as ftrain:
        with open(dev_file, "w", encoding="utf-8") as fdev:
            with open(test_file, "w", encoding="utf-8") as ftest:
                records=[]
                for line in f:
                    records.append(line)
                size=len(records)
                train_size=int(0.9*size)
                dev_size=int(0.03*size)
                test_size=size-train_size-dev_size
                print("{}/{}/{}".format(train_size, dev_size, test_size))
                for line in records[:train_size]:
                    ftrain.write(line)
                for line in records[train_size: train_size+dev_size]:
                    fdev.write(line)
                for line in records[train_size+dev_size:]:
                    ftest.write(line)
                
