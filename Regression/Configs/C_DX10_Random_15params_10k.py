import Configs.General_Configs as gConfigs

taskname = "DX10-Random-15params-10k"
log_dir = gConfigs.logs_dir + taskname
local_log_dir = "./train_logs/" + taskname

# Datasets:
training_set = "DX10-Random-15params-10k"
test_set = "TEST-DX10-Random-15params-1k"
validation_set = "VAL-DX10-Random-15params-1k"

# Hyper parameters:
n_param = 15
n_sample = 10000
batch_size = 32
max_epoches = 100
conv_friendly = True

validation_split = 0.2
