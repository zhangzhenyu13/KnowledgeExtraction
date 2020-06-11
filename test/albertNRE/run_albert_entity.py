import sys, json
import torch
import os
import numpy as np
import opennre
from opennre import encoder, model, framework
import argparse
from knowledgeextractor.nremodels import albert_encoder

parser = argparse.ArgumentParser()
parser.add_argument('--mask_entity', action='store_true', help='Mask entity mentions')
parser.add_argument("--albert_path", default="/home/zhangzy/sharedModels/pytorch_albert", help="path to the albert model")
parser.add_argument("--data_dir", default="/home/zhangzy/nlpdata/NRE", help="root data folder")
parser.add_argument("--output_dir", default="/home/zhangzy/projectModels/NRE", help="saved model ckpt dir")
parser.add_argument("--max_length", default=80, help="length of tokens per example")
args = parser.parse_args()

# Some basic settings
root_path = args.data_dir
sys.path.append(root_path)
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)
ckpt = os.path.join(args.output_dir, 'wiki80_bertentity_softmax.pth.tar')

# Check data
opennre.download('wiki80', root_path=root_path)
#opennre.download('bert_base_uncased', root_path=root_path)
rel2id = json.load(open(os.path.join(root_path, 'benchmark/wiki80/wiki80_rel2id.json')))

# Define the sentence encoder
sentence_encoder = albert_encoder.AlbertEntityEncoder(
    max_length=80, 
    pretrain_path=args.albert_path, #os.path.join(root_path, 'pretrain/bert-base-uncased'),
    mask_entity=args.mask_entity
)

# Define the model
model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)

# Define the whole training framework
framework = opennre.framework.SentenceRE(
    train_path=os.path.join(root_path, 'benchmark/wiki80/wiki80_train.txt'),
    val_path=os.path.join(root_path, 'benchmark/wiki80/wiki80_val.txt'),
    test_path=os.path.join(root_path, 'benchmark/wiki80/wiki80_val.txt'),
    model=model,
    ckpt=ckpt,
    batch_size=64, # Modify the batch size w.r.t. your device
    max_epoch=10,
    lr=2e-5,
    opt='adamw'
)

# Train the model
framework.train_model()

# Test the model
framework.load_state_dict(torch.load(ckpt)['state_dict'])
result = framework.eval_model(framework.test_loader)

# Print the result
print('Accuracy on test set: {}'.format(result['acc']))
