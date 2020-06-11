import opennre
import json
from knowledgeextractor import KGEConfig
from knowledgeextractor.nremodels import albert_encoder

class _BaseModel(object):
    def __init__(self, **args):
        self.nre_model=None

    def predict(self, query_list):
        results=[]
        for item in query_list:
            pred_and_score=self.nre_model.infer(item)
            results.append(
                pred_and_score
            )
        
        return results

class AlbertNREModel(_BaseModel):
    def __init__(self, model_config_file):
        self.config=KGEConfig(model_config_file)
        sentence_encoder=albert_encoder.AlbertEntityEncoder(
            max_length=self.config.max_length
            , pretrain_path=self.config.config_path
            , blank_padding=self.config.blank_padding
            , mask_entity=self.config.mask_entity
            , from_scratch=self.config.from_scrach
        )

        rel2id = json.load(self.config.rel2id)
        self.nre_model=opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
        self.nre_model.load_state_dict(self.config.model_dict)
    
    

class CNNNREModel(_BaseModel):
    def __init__(self, model_config_file):
        self.config=KGEConfig(model_config_file)
        word2id = json.load(self.config.word2id)
        rel2id = json.load(self.config.rel2id)

        sentence_encoder = opennre.encoder.CNNEncoder(
            token2id=word2id,
            max_length=40,
            word_size=50,
            position_size=5,
            hidden_size=230,
            blank_padding=True,
            kernel_size=3,
            padding_size=1
        )
        # Define the model
        self.nre_model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
        self.nre_model.load_state_dict(self.config.model_dict)
    
    

class PCNNAttnNREModel(_BaseModel):
    def __init__(self, model_config_file):
        self.config=KGEConfig(model_config_file)
        rel2id = json.load(self.config.rel2id)
        word2id = json.load(self.word2id)

        # Define the sentence encoder
        sentence_encoder = opennre.encoder.PCNNEncoder(
            token2id=word2id,
            max_length=120,
            word_size=50,
            position_size=5,
            hidden_size=230,
            blank_padding=True,
            kernel_size=3,
            padding_size=1
        )

        # Define the model
        self.nre_model = opennre.model.BagAttention(sentence_encoder, len(rel2id), rel2id)

        self.nre_model.load_state_dict(self.config.model_dict)

        self.nre_model.infer