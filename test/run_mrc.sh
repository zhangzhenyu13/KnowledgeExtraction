python /home/zhangzy/KnowledgeExtraction/albert/run_squad_v2.py \
   --init_checkpoint /home/zhangzy/KnowledgeExtraction/data/models/albert_base/model.ckpt-best \
   --do_train \
   --do_predict \
   --train_file /home/zhangzy/KnowledgeExtraction/data/mrc/squad/train-v2.0.json \
   --predict_file /home/zhangzy/KnowledgeExtraction/data/mrc/squad/dev-v2.0.json \
   --train_feature_file /home/zhangzy/KnowledgeExtraction/data/mrc/squad/train-features \
   --predict_feature_file /home/zhangzy/KnowledgeExtraction/data/mrc/squad/predict-features \
   --predict_feature_left_file /home/zhangzy/KnowledgeExtraction/data/mrc/squad/predict-features-left \
   --vocab_file /home/zhangzy/KnowledgeExtraction/data/models/albert_base/30k-clean.vocab \
   --albert_config_file  /home/zhangzy/KnowledgeExtraction/data/models/albert_base/albert_config.json \
   --output_dir  /home/zhangzy/KnowledgeExtraction/data/saved_models/mrc \
   #--spm_model_file home/zhangzy/KnowledgeExtraction/data/models/albert_base/30k-clean.model \

