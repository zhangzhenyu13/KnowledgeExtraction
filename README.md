## prepare

enter the repo's root dir and run "git clone https://github.com/google-research/albert.git"

then run the python setup develop to install the package  KnowledgeExtractor

## How to use the CRF Model:

### predicting phase: 

import knowledgeextractor as ke

config_file="xxxx" # repo/config/crf_albert_model.json

nermodel=ke.nermodels.crf_albert.NERModel(config_file)

query_data={
        "guid":"test1",
        "text":"this is just a test snippet!"
    }

query_list=[query_data]

results=nermodel.predict(query_list)


### training phase
training need text like 

{
"originalText":"xxxxx",

"entities":
[
    {
        
        "label_type": "疾病和诊断",

            "start_pos": 19,

            "end_pos": 27
    },
    
    ...
]
}

you can write such json string in a data.json file with each line as a json record.

run the repo/test/files_ner_file.py  and in its __main__ module specify max_sequence_length and and source file(path to data.json) and result file(where the generated train.json, deve.json, test.json are stored)

run the repo/test/run_crf.sh, where you need to prepare the init weights dict of albert chinese model(assume your model are stored in path/sharedModels/albert_base_zh/, which contains weights, vocabs, and config file for the albert).

