from knowledgeextractor.utils.chinese_CONLL import CoNLLWriter
from knowledgeextractor.utils.scores import scores
import os
from tornado import httpclient
from tornado.httputil import HTTPServerRequest
import json
from knowledgeextractor.utils import crf_utils
import random

def create_examples(file_name):
    """Creates examples from json lines."""
    lines=[]
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(line)

    examples = []
    for (i, line) in enumerate(lines):
        guid = "%s" % (i)
        data=json.loads(line)
        text=data["text"]


        token_labels= data["token_labels"]
        examples.append(
        crf_utils.InputExample(guid=guid, text=text, token_labels=token_labels))
    return examples


def eval(data, data_tag="compare-test"):
    if not os.path.exists("eval"):
        os.makedirs("eval")
    file_name = 'eval/%s' % (data_tag)
    with open(file_name, "w", encoding="utf-8") as f:
        for i in range(len(data)):
            record=data[i]
            words=record["words"]
            preds=record["preds"]
            tags=record["tags"]

            for j in range(len(words)):
                w = words[j]
                tgt = tags[j]
                pred = preds[j]
                f.write('%d %s %s %s %s %s\n' % (j + 1, w, "_", "_", tgt, pred))
    
    acc, precision, recall, f1 = scores(file_name)
    print('%s acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%%' %
         (file_name, acc, precision, recall, f1) )
    return acc, precision, recall, f1

def requestServer(query_list, bsz=8):
    
    http_client = httpclient.HTTPClient()
    query_list=[ query_list[i:i+bsz] for i in range(0,len(query_list), bsz) ]
    results=[]
    for query_batch in query_list:
        
        http_request= httpclient.HTTPRequest(
            url="http://127.0.0.1:12301/methodCore",
            method="POST",
            body=json.dumps(
                {
                    "query_list":query_batch
                }
            )
        )
        
        try:
            response = http_client.fetch(http_request)
            #print(response.body.decode(encoding="utf-8"))
            print("processed {} queries".format(len(query_batch)))
            results.extend(json.loads(response.body.decode(encoding="utf-8"))["predictions"])

        except httpclient.HTTPError as e:
            # HTTPError is raised for non-200 responses; the response
            # can be found in e.response.
            #e.with_traceback()
            print("Error: " + str(e))
        except Exception as e:
            # Other errors are possible, such as IOError.
            print("Error: " + str(e))
    http_client.close()

    return results
def loadData(data_folder, size=1000):
    
    examples=create_examples(os.path.join(data_folder,"CRF","test.json"))
    random.shuffle(examples)
    print("exmples:", len(examples))
    examples=examples[:size]
    query_list=[
        {"guid":ex.guid, "text":ex.text } for ex in examples
    ]

    return examples, query_list


if __name__ == "__main__":
    data_folder="/home/zhangzy/nlpdata"
    test_examples, test_query_list=loadData(data_folder)
    predictions=requestServer(test_query_list)

    compare_data=[
        {
            "words": predict["words"],
            "tags": t_example.token_labels,
            "preds": predict["tags"]
        } for t_example, predict in zip(test_examples,predictions)
    ]

    print(len(test_examples), len(predictions), len(compare_data))
    print(compare_data[:2])
    eval(compare_data)
