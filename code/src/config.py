#coding:utf8
import warnings
class DefaultConfig(object):
    env = 'nus-wide' 
    model = 'VFLMultiMNIST_5'
    guest_train_data_root = '/data/mnist_guest/mnist_train' 
    guest_test_data_root = '/data/mnist_guest/mnist_test' 
    host_train_data_root = '/data/mnist_host/mnist_train' 
    host_test_data_root = '/data/mnist_host/mnist_test' 
    
    guest_test_attack1_data_root = ''
    host_test_attack1_data_root = '/'

    guest_test_attack2_data_root = ''
    host_test_attack2_data_root = ''

    load_model_path = None
    batch_size = 16 # batch size
    use_gpu = True # user GPU or not
    num_workers = 8 # how many workers for loading data
    print_freq = 32 # print info every N batch

    debug_file = './debug'
    result_file = 'result.csv'

    max_epoch = 100
    lr = 1e-3 # initial learning rate
    lr_decay = 0.8
    weight_decay = 1e-4 
    beta1 = 0.9
    beta2 = 0.999

    device = 0
    mal = False
    alpha = 0.5


def parse(self,kwargs):
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" %k)
        setattr(self,k,v)

    print('user config:')
    for k,v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))


DefaultConfig.parse = parse
opt = DefaultConfig()
