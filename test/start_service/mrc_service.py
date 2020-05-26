from knowledgeextractor.nerservice.ner_server import MRCServer
import logging
logging.getLogger( __name__ ).setLevel(logging.INFO)


if __name__ == "__main__":
    server_config="/home/zhangzy/KnowledgeExtraction/config/tornado_server.json"
    mrc_model_config="/home/zhangzy/KnowledgeExtraction/config/mrc_albert_model.json"


    mrcserver=MRCServer(server_config, mrc_model_config)
    mrcserver.start()