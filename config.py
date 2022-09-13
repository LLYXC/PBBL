import os
import logging
from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment("main")
logger = logging.getLogger("debias")
ch = logging.StreamHandler()
formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%H:%M:%S"
)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel("INFO")
ex.logger = logger


@ex.config
def get_config():
    
    device = 0
    log_dir = None
    data_dir = None
    
    main_tag = None
    
    dataset_tag = None
    model_tag = None
    
    target_attr_idx = None
    bias_attr_idx = None
    
    main_num_steps = None
    main_valid_freq = None
    epochs = None
    
    main_batch_size = 256
    main_optimizer_tag = 'Adam'
    main_learning_rate = 1e-3
    main_weight_decay = 0.0
    
    main_save_logits = False
    

# User Configuration

@ex.named_config
def idle():
    idle = True

@ex.named_config
def nonidle():
    idle = False

@ex.named_config
def server_user():
    log_dir = "./log"
    data_dir = "./dataset"

# Dataset Configuration
@ex.named_config
def gender_pneumothorax(log_dir):
    dataset_tag = 'Gender_pneumothorax'
    model_tag = 'DenseNet121'
    target_attr_idx = 0
    bias_attr_idx = 1
    main_num_steps = 10 * 100
    main_valid_freq = 10
    main_batch_size = 256
    main_learning_rate = 1e-4
    main_weight_decay = 5e-4
    main_tag = 'Gender_pneumothorax'
    log_dir = os.path.join(log_dir, main_tag)

@ex.named_config
def mimic(log_dir):
    dataset_tag = 'MIMIC_CXR'
    model_tag = 'DenseNet121'
    # idx 0: pneumothorax
    # idx 1: chest tube
    target_attr_idx = 0
    bias_attr_idx = 1
    # main_num_steps = 5204 * 20
    # main_valid_freq = 5204
    # main_batch_size = 32
    main_num_steps = 34 * 40
    main_valid_freq = 34
    main_batch_size = 256
    main_learning_rate = 1e-4
    main_weight_decay = 5e-4
    main_tag = 'MIMIC_CXR'
    log_dir = os.path.join(log_dir, 'mimic')

@ex.named_config
def case1(dataset_tag, main_tag):
    dataset_tag += "_case1"
    main_tag += "_case1"

@ex.named_config
def case2(dataset_tag, main_tag):
    dataset_tag += "_case2"
    main_tag += "_case2"

@ex.named_config
def seed1(main_tag):
    main_tag += "-Seed1"

@ex.named_config
def seed2(main_tag):
    main_tag += "-Seed2"

@ex.named_config
def seed3(main_tag):
    main_tag += "-Seed3"

# Method Configuration

@ex.named_config
def adam(main_tag):
    main_optimizer_tag = "Adam"
    main_learning_rate = 1e-3
    main_weight_decay = 0
    main_tag += "_Adam"    
    
@ex.named_config
def adamw(main_tag):
    main_optimizer_tag = "AdamW"
    main_learning_rate = 1e-3
    main_weight_decay = 5e-3
    main_tag += "_AdamW"
