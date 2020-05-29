from knowledgeextractor.nerservice.tornado_http_sevrer import KGEHTTPPServer
from knowledgeextractor.nermodels import crf_albert, mrc_albert
import logging
logging.getLogger( __name__ ).setLevel(logging.INFO)

class CRFServer(KGEHTTPPServer):
    def __init__(self, server_config_file, model_config_file):
        KGEHTTPPServer.__init__(self, server_config_file)
        self.ner_model=crf_albert.NERModel(model_config_file)
    
    async def processCore(self, data):
        results=[]
        query_list= data["query_list"]
        assert type(query_list)==list
        predictions=self.ner_model.predict(query_list)
        for i, prediction in enumerate(predictions):
            pred_ids=prediction["predictions"]
            query_data=query_list[i]
            feature=self.ner_model.processor.processText(query_data)
            input_ids=feature.input_ids
            words, tags= self.ner_model.processor.recover_words_tags(pred_ids,input_ids)
            results.append({
                    "words":words,
                    "tags":tags,
                }
            )
        results={"predictions":results, "query_list":query_list}
        return results

class MRCServer(KGEHTTPPServer):
    def __init__(self, server_config_file, model_config_file):
        KGEHTTPPServer.__init__(self, server_config_file)
        self.ner_model=mrc_albert.NERModel(model_config_file)
    
    async def processCore(self, data):
        query_list= data["query_list"]
        assert type(query_list)==list
        all_nbest_json, na_probs=self.ner_model.predict(query_list)
        results=[]
        #print(all_nbest_json)
        #print(na_probs)
        for guid, entry in all_nbest_json.items():
            #print(guid)
            #print(entry)
            #print(na_probs[guid])
            entities=list(map(lambda ent: ent["text"], entry) )
            probabilities= list(map(lambda ent: ent["probability"], entry) )
            results.append(
                {
                    "guid": guid,
                    "entities": entities,
                    "probabilities": probabilities,
                    "na_prob": na_probs[guid]
                }
            )
        return results

if __name__ == "__main__":
    server_config="/home/zhangzy/KnowledgeExtraction/config/tornado_server.json"
    crf_model_config="/home/zhangzy/KnowledgeExtraction/config/crf_albert_model.json"
    mrc_model_config="/home/zhangzy/KnowledgeExtraction/config/mrc_albert_model.json"

    #crfserver=CRFServer(server_config, crf_model_config)
    #crfserver.start()

    mrcserver=MRCServer(server_config, mrc_model_config)
    mrcserver.start()