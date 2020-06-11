from knowledgeextractor.nerservice.tornado_http_sevrer import KGEHTTPPServer
from knowledgeextractor.nremodels.models import AlbertNREModel, CNNNREModel, PCNNAttnNREModel
import opennre
import logging
import json
logging.getLogger( __name__ ).setLevel(logging.INFO)

class NREAlbertServer(KGEHTTPPServer):
    def __init__(self, server_config_file, model_config_file):
        KGEHTTPPServer.__init__(self, server_config_file)
        self.nre_model=AlbertNREModel(model_config_file)

    async def processCore(self, data):
        results=[]
        query_list= data["query_list"]
        assert type(query_list)==list
        predictions=self.nre_model.predict(query_list)
        for re_name, score in predictions:
            
            results.append({
                    "relation": re_name,
                    "score": score,
                }
            )
        results={"predictions":results, "query_list":query_list}
        return results

class NRECNNServer(KGEHTTPPServer):
    def __init__(self, server_config_file, model_config_file):
        KGEHTTPPServer.__init__(self, server_config_file)
        self.nre_model=AlbertNREModel(model_config_file)

    async def processCore(self, data):
        results=[]
        query_list= data["query_list"]
        assert type(query_list)==list
        predictions=self.nre_model.predict(query_list)
        for re_name, score in predictions:
            
            results.append({
                    "relation": re_name,
                    "score": score,
                }
            )
        results={"predictions":results, "query_list":query_list}
        return results

class NREPCNNAttnServer(KGEHTTPPServer):
    def __init__(self, server_config_file, model_config_file):
        KGEHTTPPServer.__init__(self, server_config_file)
        self.nre_model=AlbertNREModel(model_config_file)

    async def processCore(self, data):
        results=[]
        query_list= data["query_list"]
        assert type(query_list)==list
        predictions=self.nre_model.predict(query_list)
        for re_name, score in predictions:
            
            results.append({
                    "relation": re_name,
                    "score": score,
                }
            )
        results={"predictions":results, "query_list":query_list}
        return results

if __name__ == "__main__":
    test_select=1
    
    server_config="/home/zhangzy/KnowledgeExtraction/config/tornado_server.json"

    if test_select ==1:
        nre_model_config="/home/zhangzy/KnowledgeExtraction/config/nre_albert_model.json"
        nre_server=NREAlbertServer(server_config, nre_model_config)
    elif test_select==2:
        nre_model_config="/home/zhangzy/KnowledgeExtraction/config/nre_cnn_model.json"
        nre_server=NRECNNServer(server_config, nre_model_config)
    elif test_select==3:
        nre_model_config="/home/zhangzy/KnowledgeExtraction/config/nre_pcnn_model.json"
        nre_server=NREPCNNAttnServer(server_config, nre_model_config)
    else:
        raise ValueError(" test select= %d is not supported!"%test_select)

    nre_server.start()