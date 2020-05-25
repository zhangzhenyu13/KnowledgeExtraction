export CUDA_VISIBLE_DEVICES=0,1
home_dir=/home/zhangzy
python ${home_dir}/KnowledgeExtraction/test/albertNER/run_crf_albert.py \
   --init_checkpoint ${home_dir}/sharedModels/albert_base_zh/model.ckpt-best \
   --albert_config_file  ${home_dir}/sharedModels/albert_base_zh/albert_config.json \
   --vocab_file ${home_dir}/sharedModels/albert_base_zh/vocab_chinese.txt \
   --max_seq_length 384 \
   --do_train \
   --do_predict \
   --do_eval \
   --data_dir ${home_dir}/nlpdata/ \
   --output_dir  ${home_dir}/projectModels/crf_ner_models/model \
   --use_gpu
