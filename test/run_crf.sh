python /home/zhangzy/KnowledgeExtraction/test/albertNER/run_crf_albert.py \
   --init_checkpoint /home/zhangzy/sharedModels/albert_base_zh/model.ckpt-best \
   --albert_config_file  /home/zhangzy/sharedModels/albert_base_zh/albert_config.json \
   --vocab_file /home/zhangzy/sharedModels/albert_base_zh/vocab_chinese.txt \
   --max_seq_length 200 \
   --do_train \
   --do_predict \
   --do_eval \
   --data_dir /home/zhangzy/nlpdata/ \
   --output_dir  /home/zhangzy/projectModels/crf_ner_models/model \

