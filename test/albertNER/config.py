import texar.tf as tx
num_epochs = 50
vocab = "/home/zhangzy/KnowledgeExtraction/data/models/albert_base_zh/vocab_chinese.txt"
label="/home/zhangzy/KnowledgeExtraction/data/ner/splitdata/labels.txt"
#data
tag_space=768
hidden_dim=768
keep_prob = 0.5
batch_size = 16
encoder = None
max_seq_length=256



opt = {
    'optimizer': {
        'type': 'AdamOptimizer',
        'kwargs': {
            'beta1': 0.9,
            'beta2': 0.997,
            'epsilon': 1e-9
        }
    }
}

lr = {
    'learning_rate_schedule': 'constant.linear_warmup.rsqrt_decay.rsqrt_depth',
    'lr_constant': 2 * (hidden_dim ** -0.5),
    'static_lr': 1e-3,
    'warmup_steps': 16000,
}