model_select=3
if [ $model_select = 1 ]; then
   python /home/zhangzy/KnowledgeExtraction/test/test_albert_ner.py \
    --model_name_or_path /home/zhangzy/KnowledgeExtraction/data/models/albert_base_zh/ \
    --config_name /home/zhangzy/KnowledgeExtraction/data/models/albert_base_zh/config.json \
    --tokenizer_name /home/zhangzy/KnowledgeExtraction/data/models/albert_base_zh/ \ #vocab_chinese.txt \
    --data_dir /home/zhangzy/KnowledgeExtraction/data/ner/COLLNER.txt \
    --labels /home/zhangzy/KnowledgeExtraction/data/ner/labels.txt \
    --output_dir /home/zhangzy/KnowledgeExtraction/data/output 
    --do_eval
elif [ $model_select = 2 ]; then
   python /home/zhangzy/KnowledgeExtraction/test/transformerNER/ner.py \
   --data_path /home/zhangzy/KnowledgeExtraction/data/ner/splitdata/ \
   --train /home/zhangzy/KnowledgeExtraction/data/ner/splitdata/train.txt \
   --dev /home/zhangzy/KnowledgeExtraction/data/ner/splitdata/dev.txt \
   --test /home/zhangzy/KnowledgeExtraction/data/ner/splitdata/test.txt 
elif [ $model_select = 3 ]; then
   python /home/zhangzy/KnowledgeExtraction/test/transformerNER/ner.py \
   --data_path /home/zhangzy/KnowledgeExtraction/data/ner/splitdata/ \
   --train /home/zhangzy/KnowledgeExtraction/data/ner/splitdata/train.txt \
   --dev /home/zhangzy/KnowledgeExtraction/data/ner/splitdata/dev.txt \
   --test /home/zhangzy/KnowledgeExtraction/data/ner/splitdata/test.txt \
   --checkpoint /home/zhangzy/KnowledgeExtraction/data/models/albert_base_zh/model \
   --albert_config  /home/zhangzy/KnowledgeExtraction/data/models/albert_base_zh/config.json 

fi