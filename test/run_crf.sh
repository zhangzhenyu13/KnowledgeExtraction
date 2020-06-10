export CUDA_VISIBLE_DEVICES=0,1
home_dir=/home/zhangzy
python ${home_dir}/KnowledgeExtraction/test/albertNER/run_crf_albert.py \
   --init_checkpoint ${home_dir}/sharedModels/albert_base_zh/model.ckpt-best \
   --albert_config_file  ${home_dir}/sharedModels/albert_base_zh/albert_config.json \
   --seq_processor ${home_dir}/KnowledgeExtraction/config/crf_processor.json \
   --max_seq_length 384 \
   --train_step 12600 \
   --do_predict \
   --do_eval \
   --data_dir ${home_dir}/nlpdata/ \
   --output_dir  ${home_dir}/projectModels/crf_ner_models/ 