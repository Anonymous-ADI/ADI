# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import copy, time, json
import base64
from collections import Counter
import sys
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
import _pickle as cPickle
from oscar.modeling.modeling_bert import ImageBertForSequenceClassification, ImageBertForSequenceClassificationOriginal, ImageBertForSequenceClassificationBP
from transformers.pytorch_transformers import WEIGHTS_NAME, BertTokenizer, BertConfig
from transformers.pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule
from oscar.utils.misc import set_seed
from oscar.utils.task_utils import (_truncate_seq_pair, convert_examples_to_features_vqa,
                        output_modes, processors)
import progressbar

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, ImageBertForSequenceClassificationBP, BertTokenizer),
}

log_json = []
debug_size = 500

class MTDatasetA2Q(Dataset):
    """ VQA Dataset """

    def __init__(self, args, tokenizer, labels, txt_root, feat_root):
        super(MTDatasetA2Q, self).__init__()
        self.args = args

        self.output_mode = output_modes[args.task_name]
        self.tokenizer = tokenizer
        self.labels = labels

        self.feat_names = os.listdir(feat_root)
        
        print('Total Images: ', len(self.feat_names))

    def raw2tensor(self, feat, Q, L,
                    cls_token_at_end=False, pad_on_left=False,
                    cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                    sequence_a_segment_id=0, sequence_b_segment_id=1,
                    cls_token_segment_id=1, pad_token_segment_id=0,
                    mask_padding_with_zero=True):
        cls_token_at_end = bool(self.args.model_type in ['xlnet']) # xlnet has a cls token at the end
        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        cls_token_segment_id = (2 if self.args.model_type in ['xlnet'] else 0)
        pad_on_left = bool(self.args.model_type in ['xlnet']) # pad on the left for xlnet
        pad_token_segment_id = (4 if self.args.model_type in ['xlnet'] else 0)

        label = torch.LongTensor([L])
        label = label.unsqueeze(1)
        label = label.expand(1, 3129)
        label = label.type(torch.FloatTensor)

        img_feat = feat
        img_feat = img_feat.type(torch.FloatTensor)
        if img_feat.shape[0] > self.args.max_img_seq_length:
            img_feat = img_feat[0:self.args.max_img_seq_length, ]
            if self.args.max_img_seq_length > 0:
                input_mask_img = [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
        else:
            if self.args.max_img_seq_length > 0:
                input_mask_img = [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
            padding_matrix = torch.zeros((self.args.max_img_seq_length - img_feat.shape[0], img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)
            if self.args.max_img_seq_length > 0:
                input_mask_img = input_mask_img + ([0 if mask_padding_with_zero else 1] * padding_matrix.shape[0])

        tokens_a = self.tokenizer.tokenize(Q)

        if len(tokens_a) > self.args.max_seq_length - 2:
            tokens_a = tokens_a[:(self.args.max_seq_length - 2)]

        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        padding_length = self.args.max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == self.args.max_seq_length
        assert len(input_mask) == self.args.max_seq_length
        assert len(segment_ids) == self.args.max_seq_length

        input_mask = input_mask + input_mask_img
        
        return (
            img_feat.unsqueeze(0),
            label,
            torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
            torch.tensor(input_mask, dtype=torch.long).unsqueeze(0),
            torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0)
        )

    def __getitem__(self, index):
        return self.feat_names[index]

    def __len__(self):
        return len(self.Q_names)

eval_dataset = None
label2ans = None

class GradSaver:
    def __init__(self):
        self.grad = -1
    
    def save_grad(self, grad):
        self.grad = grad

def BP_multi(model, Q_list, F, L, mask_tensor=None, no_label=False):
    global eval_dataset
    global label2ans
    model.train()
    for i in range(len(Q_list)):
        Q = Q_list[i]
        (img_feats, label, input_ids, input_mask, segment_ids) = eval_dataset.raw2tensor(F, Q, L)

        img_feats.requires_grad = True

        inputs = {'input_ids':      input_ids,
                'attention_mask': input_mask,
                'token_type_ids': segment_ids,
                'labels':         label,
                'img_feats':      img_feats,
                'no_label':       no_label}
        model.zero_grad()
        outputs = model(**inputs)
        loss, logits = outputs[:2]
        val, idx = logits.max(1)
        pred_L = idx[0].item()
        word_inputs = outputs[-1]
        loss.backward(retain_graph=True)
        grad1 = img_feats.grad.detach().clone()
        img_feats.grad.data.zero_()
        model.zero_grad()
        grad2 = torch.autograd.grad(torch.var(logits), word_inputs, create_graph=True, retain_graph=True)
        grad3 = torch.autograd.grad(torch.abs(grad2[0]).mean(), img_feats, retain_graph=True)
        if i == 0:
            img_feats_grad = 0.7 * grad1[0] / len(Q_list) + 0.3 * grad3[0][0] / len(Q_list)
        else:
            img_feats_grad += 0.7 * grad1[0] / len(Q_list) + 0.3 * grad3[0][0] / len(Q_list)

    img_feats_grad = img_feats_grad[:F.shape[0]]
    assert img_feats_grad.shape == F.shape
    return img_feats_grad, pred_L

def infer(model, Q, F, L, no_label=False):
    global eval_dataset
    global label2ans
    with torch.no_grad():
        model.eval()
        (img_feats, label, input_ids, input_mask, segment_ids) = eval_dataset.raw2tensor(F, Q, L)
        inputs = {'input_ids':      input_ids,
                  'attention_mask': input_mask,
                  'token_type_ids': segment_ids,
                  'labels':         None,
                  'img_feats':      img_feats,
                  'no_label':       no_label}
        outputs = model(**inputs)
        logits = outputs[0]
        val, idx = logits.max(1)
        pred_L = idx[0].item()
        pred_answer = label2ans[eval_dataset.labels[pred_L]]
        target_answer = label2ans[eval_dataset.labels[L]]
        return pred_L, pred_answer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ID", default=-1, type=int, help="ID")
    parser.add_argument("--Q_type", default='add', type=str, help="Q_type")
    parser.add_argument("--is_diff", default=1, type=int, help="Q_type")
    parser.add_argument("--F_type", default='masked', type=str, help="Q_type")
    parser.add_argument("--Q_root", default='', type=str, help='Q_root')
    parser.add_argument('--F_root', default='', type=str, help='F_root')
    parser.add_argument("--data_dir", default='/data/COCO/REMOVE/feature/', type=str, required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--txt_data_dir", default='/data/COCO/REMOVE/question/', type=str, required=False,
                        help="The input text data dir. Should contain the .json files (or other data files) for the task.")
    
    parser.add_argument("--model_name_or_path", default='/collection/pretrained-models/Oscar/base_best/', type=str, required=False,
                        help="Path to pre-trained model or shortcut name")
    
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int, help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument("--output_dir", default='/output/MT-VQA/Oscar/', type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--label_file", type=str, default='/collection/pretrained-models/Oscar/trainval_ans2label.pkl', help="Label Dictionary")

    parser.add_argument("--per_gpu_train_batch_size", default=256, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_val_batch_size", default=32, type=int, help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train_val", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run test on the test set.")
    parser.add_argument("--do_test_dev", action='store_true', help="Whether to run test on the test-dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true', help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")

    parser.add_argument("--model_type", default='bert', type=str, required=False,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--task_name", default='vqa_text', type=str, required=False,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    
    parser.add_argument("--label2ans_file", type=str, default='/collection/pretrained-models/Oscar/trainval_label2ans.pkl', help="Label to Answer Dictionary")

    parser.add_argument("--img_feat_dir", default=None, type=str, help="The input img_feat_dir.")
    parser.add_argument("--img_feat_format", default='pt', type=str, help="img_feat_format: pt or tsv.")

    parser.add_argument("--data_label_type", default='mask', type=str, help="faster or mask")
    parser.add_argument("--loss_type", default='bce', type=str, help="kl or xe")
    parser.add_argument("--use_vg", action='store_true', help="Use VG-QA or not.")
    parser.add_argument("--use_vg_dev", action='store_true', help="Use VG-QA as validation.")

    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    
    parser.add_argument("--drop_out", default=0.3, type=float, help="Drop out for BERT.")
    parser.add_argument("--adjust_dp",action='store_true', help="Adjust Drop out for BERT.")

    parser.add_argument("--adjust_loss", action='store_true', help="Adjust Loss Type for BERT.")
    parser.add_argument("--adjust_loss_epoch", default=-1, type=int, help="Adjust Loss Type for BERT.")
    parser.add_argument("--classifier", default='linear', type=str, help="linear or mlp")
    parser.add_argument("--cls_hidden_scale", default=3, type=int, help="cls_hidden_scale: for classifier")

    parser.add_argument("--hard_label", action='store_true', help="Soft Label or Hard Label.")

    parser.add_argument("--max_img_seq_length", default=50, type=int, help="The maximum total input image sequence length.")
    parser.add_argument("--img_feature_dim", default=2054, type=int, help="The Image Feature Dimension.")
    parser.add_argument("--img_feature_type", default='faster_r-cnn', type=str, help="faster_r-cnn or mask_r-cnn")
    parser.add_argument("--code_voc", default=512, type=int, help="dis_code_voc: 256, 512")
    parser.add_argument("--code_level", default='top', type=str, help="code level: top, botttom, both")

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=25, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=-1, help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_epoch', type=int, default=5, help="Save checkpoint every X epochs.")
    parser.add_argument('--save_after_epoch', type=int, default=-1, help="Save checkpoint after epoch.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true', help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true', help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=88, help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    parser.add_argument("--philly", action='store_true', help="Use Philly: reset the output dir")
    parser.add_argument("--load_fast", action='store_true', help="Load Tensor Fast")
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--num_pics', default=50, type=int)

    parser.add_argument('--mutate', default='unbounded', type=str)

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train: logger.info("Output Directory Exists.")

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    set_seed(args.seed, args.n_gpu)
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))

    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels(args.label_file)
    num_labels = len(label_list)
    logger.info('Task Name: {}, #Labels: {}'.format(args.task_name, num_labels))

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels, finetuning_task=args.task_name,
    )
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    config.img_feature_dim = args.img_feature_dim
    config.img_feature_type = args.img_feature_type
    config.code_voc = args.code_voc
    config.hidden_dropout_prob = args.drop_out
    config.loss_type = args.loss_type
    config.classifier = args.classifier
    config.cls_hidden_scale = args.cls_hidden_scale

    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    model.to(args.device)
    model = nn.DataParallel(model).cuda()
    
    global eval_dataset
    global label2ans
    eval_dataset = MTDatasetA2Q(args, tokenizer, label_list,
                            txt_root=args.txt_data_dir,
                            feat_root=args.data_dir)

    label2ans = cPickle.load(open(args.label2ans_file, 'rb'))

    ans2label = {}
    for k, v in enumerate(label2ans):
        ans2label[v] = k

    label2target = {}
    for i, label in enumerate(label_list):
        label2target[label] = i
    
    sys.path.insert(0, '/code/ADV-VQA/')
    from Adversary import AdvImageToFeat
    import cv2
    import json
    import random

    data_path = '/data/BUTD-config/data/genome/1600-400-20'
    yaml_path = '/data/BUTD-config/configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml'
    weights_path = '/collection/pretrained-models/BUTD/faster_rcnn_from_caffe_attr.pkl'

    Adv = AdvImageToFeat(
            data_path=data_path,
            yaml_path=yaml_path,
            weights_path=weights_path
        )

    exp_name = 'test_mask'

    QA_json_path = '/data/VQA/val2014_image_as_key.json'
    val2014_I_dir = '/data/COCO/val2014/'
    # output_path = '/output/ADV/%s/images/' % exp_name  
    image_list = sorted(os.listdir(val2014_I_dir))

    with open(QA_json_path, 'r') as f:
        QA_dic = json.load(f)

    total = 0
    succ = 0
    json_record = {}
    pred_A_dic = {}
    
    image_list = image_list[:args.num_pics]
    def dominated_rate(model, Q_list, F, L):
        pred_list = []
        total_cnt = 0.0
        dominated_cnt = 0.0
        for Q in Q_list:
            pred_L, pred_answer = infer(model, Q, F, L)
            total_cnt += 1.0
            if pred_L == L:
                dominated_cnt += 1.0
        return dominated_cnt / total_cnt

    image_item_list = []
    progress = progressbar.ProgressBar(maxval=len(image_list)).start()
    for i, image_name in enumerate(image_list):
        progress.update(i + 1)
        image_path = val2014_I_dir + image_name
        raw_image = cv2.imread(image_path)
        raw_image = cv2.resize(raw_image, (128, 128))
        
        _, BUTD_feat, oscar_feat = Adv.doit_infer(raw_image)
        F = oscar_feat
        
        image_id = str(int(image_name.split('.')[0].split('_')[-1]))
        Q_list = QA_dic[image_id]['Q']
        A_list = []
        for Q in Q_list:
            pred_L, pred_answer = infer(model, Q, F, 0)
            A_list.append(pred_L)
        
        A_dict = Counter(A_list)
        L = max(A_dict, key = A_dict.get)
        
        acc = dominated_rate(model, Q_list, F, L)
        image_item_list.append({
            'image_name': image_name,
            'image_data': raw_image,
            'acc': acc,
            'times': 0,
            'target_A': L
            })
    progress.finish()

    record_cnt = 0
    for image_item in image_item_list:
        if image_item['acc'] == 1:
            record_cnt += 1
    print('Percentage of Dominated Image: %f' % (record_cnt / len(image_item_list)))

    def sort_key(image_item):
        return image_item['acc']

    image_item_list = sorted(image_item_list, key=sort_key, reverse=True)

    output_list = []

    def get_mask(raw_image, mask, Q, L, no_label, update_epoch=1):
        for k in range(update_epoch):
            _, BUTD_feat, oscar_feat, mask_tensor = Adv.doit(raw_image, 
                                                at_first=False,
                                                mask=mask)
            F = oscar_feat
            grad, pred_L = BP(model, Q, F, L, mask_tensor, no_label=no_label)
            if pred_L == L:
                break
            mutated_mask = Adv.mutate_mask(grad, 100.0)
            mask = mutated_mask
            mask = np.clip(mask, a_min=0, a_max=1)
        print(np.mean(mask))
        return mask, pred_L

    exp_length = args.num_pics
    
    progress = progressbar.ProgressBar(maxval=exp_length).start()
    for _iter in range(exp_length):
        progress.update(_iter)
        idx = 0
        image_item = image_item_list[idx]

        if image_item['acc'] == 1:
            output_list.append(image_item)
            print(image_item['image_name'])
            del image_item_list[idx]
            continue
        image_name = image_item['image_name']
        image_id = str(int(image_name.split('.')[0].split('_')[-1]))
        Q_list = QA_dic[image_id]['Q']
        A_list = QA_dic[image_id]['A']
        L = image_item['target_A']

        raw_image = image_item['image_data']
        _, _, oscar_feat = Adv.doit(raw_image,
                                        at_first=True,
                                        mask=None)
        F = oscar_feat
        success_cnt = 0

        min_F = F.min()
        max_F = F.max()
        orig_F = F.clone().detach()
        for iters in range(30):
            grad, pred_L = BP_multi(model, Q_list, F, L, mask_tensor=None, no_label=False)
            if args.mutate == 'unbounded':
                F -= 0.01 * grad.cuda()
            else:
                F -= 1e-7 * grad.cuda()
                # F = orig_F + (F - orig_F) / torch.abs(F - orig_F).max() * 2.0
                F = torch.clamp(F, min_F, max_F)

            if dominated_rate(model, Q_list, F, L) == 1:
                break

        new_d = dominated_rate(model, Q_list, F, L)
        print('new d: ', new_d)
        if new_d == 1:
            image_item['acc'] = 1
            output_list.append(image_item)
            del image_item_list[idx]
            print(' Found: %s' % image_item['image_name'])

        del image_item_list[idx]
        del orig_F, F
        print(' Found:', len(output_list))
    progress.finish() 

if __name__ == "__main__":
    main()
