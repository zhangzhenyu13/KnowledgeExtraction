from knowledgeextractor.nerservice.ner_server import CRFServer
import logging
logging.getLogger( __name__ ).setLevel(logging.INFO)


if __name__ == "__main__":
    server_config="/home/zhangzy/KnowledgeExtraction/config/tornado_server.json"
    crf_model_config="/home/zhangzy/KnowledgeExtraction/config/crf_albert_model.json"

    crfserver=CRFServer(server_config, crf_model_config)
    crfserver.start()
