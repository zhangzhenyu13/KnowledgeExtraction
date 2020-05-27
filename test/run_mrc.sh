export CUDA_VISIBLE_DEVICES=0,1
home_dir=/home/zhangzy

python ${home_dir}/KnowledgeExtraction/albert/run_squad_v2.py \
   --init_checkpoint ${home_dir}/sharedModels/albert_base_zh/model.ckpt-best \
   --albert_config_file  ${home_dir}/sharedModels/albert_base_zh/albert_config.json \
   --vocab_file ${home_dir}/sharedModels/albert_base_zh/vocab_chinese.txt \
   --do_train \
   --do_predict \
   --max_seq_length 256 \
   --doc_stride 64 \
   --max_query_length 64 \
   --train_file ${home_dir}/nlpdata/MRC/train.json \
   --predict_file ${home_dir}/nlpdata/MRC/dev.json \
   --train_feature_file ${home_dir}/nlpdata/MRC/train-features \
   --predict_feature_file ${home_dir}/nlpdata/MRC/predict-features \
   --predict_feature_left_file ${home_dir}/nlpdata/MRC/predict-features-left \
   --output_dir  ${home_dir}/projectModels/mrc_ner_models/ \

