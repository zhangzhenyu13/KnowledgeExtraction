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

run the repo/test/crf_gen_taggers.py  and in its __main__ module specify max_sequence_length and and source file(path to data.json) and then run the split_files.py in same pathto gen result file(where the generated train.json, deve.json, test.json are stored)

run the repo/test/run_crf.sh, where you need to prepare the init weights dict of albert chinese model(assume your model are stored in path/sharedModels/albert_base_zh/, which contains weights, vocabs, and config file for the albert).

# Sytem Deployment is now available
## specify the config file
#### tornado_server.json
#### crf_processor.json
#### crf_albert_model.json

## run the repo/test/start_service/crf_service.py

### example of request using post method
#### url should be like this
http://ip:port/methodCore (port is specified in tornado_server.json and ip the machine's IP that runs the service)

#### post body should be a json string like this
{
    "query_list":[

        {"guid":"id1 str",
        "text":"a test text-1."},

        {"guid":"id2 str",
        "text":"a test text-2."}
    ]
}

#### the returned results contains
{
    "predictions":[

        {"words":["list of words of text-1"],

        "tags":["list of tags for each word]},

        {
        "words":["list of words of text-2"],

        "tags":["list of tags for each word]
        }
    ],

    "query_list":[....(the query list defined above)]
}