import random
from pathlib import Path
import os
import sys
import json
import torch
import numpy as np
import warnings
import torch.nn.functional as F
import torch.nn as nn
from collections import Counter

from tqdm import tqdm, trange
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

warnings.filterwarnings('ignore')

# project_path = os.path.abspath('')
# if project_path not in sys.path:
#     sys.path.append(project_path)
sys.path.append("/mnt/xinfeng/research/AI_Human_Detection/Test_Ruoxin_Wang/SeqXGPT/SeqXGPT/")
import backend_model_info
from dataloader import DataManager
from model import ModelWiseCNNClassifier, ModelWiseTransformerClassifier, TransformerOnlyClassifier
from model import SeqXGPTModel


class SupervisedTrainer:
    def __init__(self, data, model, en_labels, id2label, args):
        self.data = data
        self.model = model
        self.en_labels = en_labels
        # id2label: Mapping from IDs to labels.id2label.
        self.id2label =id2label

        self.seq_len = args.seq_len
        self.num_train_epochs = args.num_train_epochs
        self.weight_decay = args.weight_decay
        self.lr = args.lr
        # Warm-up ratio for the learning rate scheduler.
        # specifies the fraction of the total training steps during which the learning rate will gradually increase (warm-up phase).
        # After the warm-up phase, the learning rate typically decays according to the scheduler.
        # 1. Stabilizing Training:
        # At the beginning of training, large gradients can destabilize updates.
        # A warm-up period allows the model to start training with small learning rates and gradually ramp up.

        # 2. Preventing Gradient Explosions:
        # Gradually increasing the learning rate reduces the risk of large updates that might cause gradient explosions.
        
        # 3. Improved Convergence:
        # Warm-up has been shown empirically to improve the convergence of large-scale models and transformers.
        self.warm_up_ratio = args.warm_up_ratio

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.model.to(self.device)
        self._create_optimizer_and_scheduler()

    def _check_data_balance(self):
        all_labels = []
        for batch in self.data.train_dataloader:
            all_labels.extend(batch['labels'].cpu().numpy().flatten())
        unique, counts = np.unique(all_labels, return_counts=True)
        # print("Label distribution:", dict(zip(unique, counts)))

    def _create_optimizer_and_scheduler(self):
        num_training_steps = len(
            self.data.train_dataloader) * self.num_train_epochs
        no_decay = ["bias", "LayerNorm.weight"]

        # Retrieve model parameters.
        named_parameters = self.model.named_parameters()
        # Separate parameters into decay and no-decay groups.
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in named_parameters
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                self.weight_decay,
            },
            {
                "params": [
                    p for n, p in named_parameters
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0,
            },
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.lr,
            betas=(0.9, 0.98),
            eps=1e-8,
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warm_up_ratio * num_training_steps,
            num_training_steps=num_training_steps)

    def load_model(self, ckpt_name):
        """Safe model loading with proper allowlisting"""
        from torch.serialization import default_restore_location
        
        # Allowlist custom classes
        torch.serialization.add_safe_globals([SeqXGPTModel])
        
        # Load with weights_only=True
        state_dict = torch.load(
            ckpt_name,
            map_location=lambda storage, loc: default_restore_location(storage, str(loc)),
            weights_only=True
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
    
    def train(self, ckpt_name='linear_en.pt'):
        for epoch in trange(int(self.num_train_epochs), desc="Epoch"):
            self.model.train()
            # Accumulate training loss.
            tr_loss = 0
            # Count training steps.
            nb_tr_steps = 0
            # Verify data balance.
            self._check_data_balance()
            # train
            for step, inputs in enumerate(
                    tqdm(self.data.train_dataloader, desc="Iteration")):
                for k, v in inputs.items():
                    # Move tensors to the appropriate device.
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                        
                # Enable gradient calculation.
                with torch.set_grad_enabled(True):
                    labels = inputs['labels']
                    output = self.model(inputs['features'], inputs['labels'])
                    logits = output['logits']
                    loss = output['loss']
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    tr_loss += loss.item()
                    nb_tr_steps += 1

            # Compute and log the average training loss.
            loss = tr_loss / nb_tr_steps
            print(f'epoch {epoch+1}: train_loss {loss}')
            # test
            self.test()
            print('*' * 120)
            torch.save(self.model.cpu(), ckpt_name)
            self.model.to(self.device)
            #################
            torch.save(self.model.state_dict(), ckpt_name)
            ###############

        print("Training finished. Loading best model...")
        ckpt_path = os.path.abspath(ckpt_name)
        ##########
        torch.save(self.model.state_dict(), ckpt_path)
        #########
        # torch.save(self.model.cpu(), ckpt_name)
        # saved_model = torch.load(ckpt_name, weights_only=True)
        # self.model.load_state_dict(saved_model.state_dict())
        return

    def test(self, content_level_eval=False):
        self.model.eval()
        texts = []
        true_labels = []
        pred_labels = []
        total_logits = []

        for step, inputs in enumerate(tqdm(self.data.test_dataloader, desc="Testing")):
            # Move inputs to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in inputs.items()}
            # Disable gradient calculation.
            with torch.no_grad():
                output = self.model(inputs['features'], inputs['labels'])
                # print("Processed labels:", output['proc_labels'][0][:10])
                # print("Predictions:", output['preds'][0][:10])
                logits = output['logits']
                proc_labels = output.get('proc_labels', inputs['labels'])
                
                texts.extend(inputs['text'])
                pred_labels.extend(output['preds'].cpu().tolist())
                true_labels.extend(proc_labels.cpu().tolist())
                total_logits.extend(logits.cpu().tolist())
        
        ######################
        # Count unique predictions
        true_counts = Counter([label for seq in true_labels for label in seq if label != -1])
        pred_counts = Counter([label for seq in pred_labels for label in seq if label != -1])

        print("\n🔍 **True Labels Distribution**:", true_counts)
        print("🔍 **Predicted Labels Distribution**:", pred_counts)
        ######################

        # with open("", 'w') as f:
        #     f.write(json.dumps(total_logits[3], ensure_ascii=False) + '\n')
        #     f.write(json.dumps(texts[3], ensure_ascii=False) + '\n')
        #     f.write(json.dumps(true_labels[3], ensure_ascii=False) + '\n')
        #     f.write(json.dumps(pred_labels[3], ensure_ascii=False) + '\n')

        # Convert to numpy arrays
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)
        
        # Evaluate at different levels.
        if content_level_eval:
            # content level evaluation
            print("*" * 8, "Content Level Evalation", "*" * 8)
            content_result = self.content_level_eval(texts, true_labels, pred_labels)
        else:
            # sent level evalation
            print("*" * 8, "Sentence Level Evalation", "*" * 8)
            sent_result = self.sent_level_eval(texts, true_labels, pred_labels)

        # Word-level evaluation with improved handling
        print("\n" + "*" * 8 + " Word Level Evaluation " + "*" * 8)
        total_valid = 0
        correct = 0
        all_true = []
        all_pred = []
        
        for t_seq, p_seq in zip(true_labels, pred_labels):
            # Convert to numpy arrays and filter padding
            t_seq = np.array(t_seq)
            p_seq = np.array(p_seq)
            valid_mask = t_seq != -1
            
            valid_t = t_seq[valid_mask]
            valid_p = p_seq[valid_mask]
            
            # Safety checks
            if len(valid_t) == 0:
                continue
                
            if len(valid_t) != len(valid_p):
                print(f"Length mismatch: true {len(valid_t)} vs pred {len(valid_p)}")
                continue
                
            total_valid += len(valid_t)
            correct += (valid_t == valid_p).sum()
            all_true.extend(valid_t.tolist())
            all_pred.extend(valid_p.tolist())

        # Calculate metrics
        if total_valid > 0:
            accuracy = correct / total_valid
            macro_f1 = f1_score(all_true, all_pred, average='macro')
            print(f"Word Level Accuracy: {accuracy*100:.1f}%")
            print(f"Macro F1 Score: {macro_f1*100:.1f}%")
            print(f"Total Valid Tokens: {total_valid:,}")
            
            # Per-class metrics
            unique_labels = np.unique(all_true)
            if len(unique_labels) > 1:
                precision = precision_score(all_true, all_pred, average=None)
                recall = recall_score(all_true, all_pred, average=None)
                print("\nClass-wise Performance:")
                for idx, label in enumerate(unique_labels):
                    label_name = self.id2label.get(label, str(label))
                    print(f"{label_name}: Precision {precision[idx]*100:.1f}% | Recall {recall[idx]*100:.1f}%")
        else:
            print("No valid tokens found for evaluation!")

        return {
            'true_labels': all_true,
            'pred_labels': all_pred,
            'total_valid': total_valid
        }
    
    def content_level_eval(self, texts, true_labels, pred_labels):
        from collections import Counter

        true_content_labels = []
        pred_content_labels = []
        for text, true_label, pred_label in zip(texts, true_labels, pred_labels):
            true_label = np.array(true_label)
            pred_label = np.array(pred_label)
            mask = true_label != -1
            true_label = true_label[mask].tolist()
            pred_label = pred_label[mask].tolist()
            true_common_tag = self._get_most_common_tag(true_label)
            true_content_labels.append(true_common_tag[0])
            pred_common_tag = self._get_most_common_tag(pred_label)
            pred_content_labels.append(pred_common_tag[0])
        
        true_content_labels = [self.en_labels[label] for label in true_content_labels]
        pred_content_labels = [self.en_labels[label] for label in pred_content_labels]
        result = self._get_precision_recall_acc_macrof1(true_content_labels, pred_content_labels)
        return result

    def sent_level_eval(self, texts, true_labels, pred_labels):
        """
        """
        true_sent_labels = []
        pred_sent_labels = []
        for text, true_label, pred_label in zip(texts, true_labels, pred_labels):
            true_sent_label = self.get_sent_label(text, true_label)
            pred_sent_label = self.get_sent_label(text, pred_label)
            true_sent_labels.extend(true_sent_label)
            pred_sent_labels.extend(pred_sent_label)
        
        true_sent_labels = [self.en_labels[label] for label in true_sent_labels]
        pred_sent_labels = [self.en_labels[label] for label in pred_sent_labels]
        result = self._get_precision_recall_acc_macrof1(true_sent_labels, pred_sent_labels)
        return result

    def get_sent_label(self, text, label):
        import nltk
        sent_separator = nltk.data.load('tokenizers/punkt/english.pickle')
        sents = sent_separator.tokenize(text)

        offset = 0
        sent_label = []
        for sent in sents:
            start = text[offset:].find(sent) + offset
            end = start + len(sent)
            offset = end
            
            split_sentence = self.data.split_sentence
            end_word_idx = len(split_sentence(text[:end]))
            if end_word_idx > self.seq_len:
                break
            word_num = len(split_sentence(text[start:end]))
            start_word_idx = end_word_idx - word_num
            tags = label[start_word_idx:end_word_idx]
            most_common_tag = self._get_most_common_tag(tags)
            sent_label.append(most_common_tag[0])
        
        if len(sent_label) == 0:
            print("empty sent label list")
        return sent_label
    
    def _get_most_common_tag(self, tags):
        """most_common_tag is a tuple: (tag, times)"""
        from collections import Counter

        # Filter out padding (-1) tags.
        filtered_tags = [t for t in tags if t != -1]
        if not filtered_tags:
            # Return a default tag. Adjust the default to match one of your keys.
            return ("human", 1)  # or ("gpt2", 1) based on your application
        mapped_tags = [self.id2label[tag] for tag in filtered_tags]
        mapped_tags = [tag.split('-')[-1] for tag in mapped_tags]
        tag_counts = Counter(mapped_tags)
        most_common_tag = tag_counts.most_common(1)[0]
        return most_common_tag

    def _get_precision_recall_acc_macrof1(self, true_labels, pred_labels):
        accuracy = accuracy_score(true_labels, pred_labels)
        macro_f1 = f1_score(true_labels, pred_labels, average='macro')
        print("Accuracy: {:.1f}".format(accuracy*100))
        print("Macro F1 Score: {:.1f}".format(macro_f1*100))

        precision = precision_score(true_labels, pred_labels, average=None)
        recall = recall_score(true_labels, pred_labels, average=None)
        print("Precision/Recall per class: ")
        precision_recall = ' '.join(["{:.1f}/{:.1f}".format(p*100, r*100) for p, r in zip(precision, recall)])
        print(precision_recall)

        result = {"precision":precision, "recall":recall, "accuracy":accuracy, "macro_f1":macro_f1}
        return result


def construct_bmes_labels(labels):
    prefix = ['B-', 'M-', 'E-', 'S-']
    id2label = {}
    counter = 0

    for label, id in labels.items():
        for pre in prefix:
            id2label[counter] = pre + label
            counter += 1
    
    return id2label

# def split_dataset(data_path, train_path, test_path, train_ratio=0.9):
#     file_names = [file_name for file_name in os.listdir(data_path) if file_name.endswith('.jsonl')]
#     print('*'*32)
#     print('The overall data sources:')
#     print(file_names)
#     file_paths = [os.path.join(data_path, file_name) for file_name in file_names]

#     total_samples = []
#     for file_path in file_paths:
#         with open(file_path, 'r') as f:
#             samples = [json.loads(line) for line in f]
#             total_samples.extend(samples)
    
#     import random
#     random.seed(0)
#     random.shuffle(total_samples)

#     split_index = int(len(total_samples) * train_ratio)
#     train_data = total_samples[:split_index]
#     test_data = total_samples[split_index:]

#     def save_dataset(fpath, data_samples):
#         with open(fpath, 'w', encoding='utf-8') as f:
#             for sample in tqdm(data_samples):
#                 f.write(json.dumps(sample, ensure_ascii=False) + '\n')
#     save_dataset(train_path, train_data)
#     save_dataset(test_path, test_data)
#     print()
#     print("The number of train dataset:", len(train_data))
#     print("The number of test  dataset:", len(test_data))
#     print('*'*32)
#     pass

def split_dataset(data_path, train_path, test_path, train_ratio=0.9, seed=42):
    """
    Splits the dataset into training and test sets ensuring no overlap.
    
    :param data_path: Path to the directory containing dataset files.
    :param train_path: Path to save the training dataset.
    :param test_path: Path to save the test dataset.
    :param train_ratio: Fraction of data to use for training (default: 0.9).
    :param seed: Random seed for reproducibility.
    """
    random.seed(seed)
    
    # Collect all JSONL files in the data directory
    file_names = [file for file in os.listdir(data_path) if file.endswith('.jsonl')]
    print('*' * 32)
    print('The overall data sources:', file_names)

    total_samples = []

    # Load data from each JSONL file
    for file_name in file_names:
        file_path = os.path.join(data_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            samples = [json.loads(line) for line in f]
            total_samples.extend(samples)
    
    # Shuffle the dataset
    random.shuffle(total_samples)

    # Get unique texts to prevent duplication
    text_to_sample = {sample['text']: sample for sample in total_samples}
    unique_samples = list(text_to_sample.values())

    # Perform the train/test split
    split_index = int(len(unique_samples) * train_ratio)
    train_data = unique_samples[:split_index]
    test_data = unique_samples[split_index:]

    def save_dataset(fpath, data_samples):
        """Helper function to save datasets in JSONL format."""
        with open(fpath, 'w', encoding='utf-8') as f:
            for sample in tqdm(data_samples, desc=f"Saving {Path(fpath).stem}"):
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    # Save datasets
    save_dataset(train_path, train_data)
    save_dataset(test_path, test_data)

    print("\n✅ Dataset split completed!")
    print(f"🚀 Train samples: {len(train_data)}")
    print(f"🚀 Test samples: {len(test_data)}")

    # Verify no overlap
    train_texts = {sample['text'] for sample in train_data}
    test_texts = {sample['text'] for sample in test_data}
    overlap = train_texts.intersection(test_texts)

    print(f"🔍 Overlap between train and test: {len(overlap)} samples (should be 0)")
    assert len(overlap) == 0, "❌ Data leakage detected! Overlapping samples found in train and test sets."
    print("✅ No data leakage! Train and test sets are properly split.")

    print('*' * 32)

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    # Add argument for processing method selection.
    parser.add_argument("--method", type=str, choices=["patch_average", "convolution_like", "patch_shuffle"], default="patch_shuffle")
    parser.add_argument("--patch_size", type=int, default=10)
    parser.add_argument("--kernel_size", type=int, default=10)
    parser.add_argument("--stride", type=int, default=5)
    #=============================================#

    parser.add_argument('--model', type=str, default='SeqXGPT')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--train_mode', type=str, default='classify')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=1024)

    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--split_dataset', action='store_true')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--train_path', type=str, default='')
    parser.add_argument('--test_path', type=str, default='')

    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--warm_up_ratio', type=float, default=0.1)

    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--test_content', action='store_true')
    return parser.parse_args()

# python ./Seq_train/train.py --gpu=0 --split_dataset
# python ./Seq_train/train.py --gpu=0
if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.split_dataset:
        print("Log INFO: split dataset...")
        split_dataset(data_path=args.data_path, train_path=args.train_path, test_path=args.test_path, train_ratio=args.train_ratio)

    # en_labels = backend_model_info.en_labels
    # en_labels = {
    #     'gpt2': 0,
    #     'gptneo': 1,
    #     'gptj': 2,
    #     'llama': 3,
    #     'gpt3re': 4,
    #     # 'gpt3sum': 3,
    #     'human': 5
    # }
    # en_labels = {'AI':0, 'human':1}
    en_labels = {
        'gpt2': 0,
        'human': 1,
    }

    id2label = construct_bmes_labels(en_labels)
    label2id = {v: k for k, v in id2label.items()}

    data = DataManager(
                    train_path=args.train_path, 
                    test_path=args.test_path, 
                    batch_size=args.batch_size, 
                    max_len=args.seq_len, 
                    human_label='human', 
                    id2label=id2label
                    )
    
    """linear classify"""
    if args.train_mode == 'classify':
        print('-' * 32 + 'classify' + '-' * 32)
        if args.model == 'SeqXGPT':
            classifier = SeqXGPTModel(embedding_size=512, seq_len=1024, num_layers=4, num_heads=2, id2labels=id2label)
            classifier.to("cuda" if torch.cuda.is_available() else "cpu")
            ckpt_name = 'seqxgpt_cls_model.pt'
        elif args.model == 'CNN':
            print('-' * 32 + "CNN" + '-' * 32)
            classifier = ModelWiseCNNClassifier(id2labels=id2label)
            ckpt_name = 'cnn_cls_model.pt'
        elif args.model == 'RNN':
            print('-' * 32 + "RNN" + '-' * 32)
            classifier = TransformerOnlyClassifier(id2labels=id2label, seq_len=args.seq_len)
            ckpt_name = 'rnn_cls_model.pt'
        else:
            classifier = ModelWiseTransformerClassifier(id2labels=id2label, seq_len=args.seq_len)
            ckpt_name = 'transformer_cls_model.pt'

        trainer = SupervisedTrainer(data, classifier, en_labels, id2label, args)

        if args.do_test:    
            print("Log INFO: do test...")
            classifier = SeqXGPTModel(embedding_size=512, seq_len=1024, num_layers=4, num_heads=2, id2labels=id2label)
            #########################
            trainer = SupervisedTrainer(data, classifier, en_labels, id2label, args)
            trainer.load_model(ckpt_name)
            trainer.test(content_level_eval=args.test_content)
            ###########################
            # saved_model = torch.load(ckpt_name, weights_only=True)
            # trainer.model.load_state_dict(saved_model.state_dict())
            # trainer.test(content_level_eval=args.test_content)
        else:
            print("Log INFO: do train...")
            trainer.train(ckpt_name=ckpt_name)
            #########
            trainer.load_model(ckpt_name)
            #########

    """contrastive training"""
    if args.train_mode == 'contrastive_learning':
        print('-' * 32 + 'contrastive_learning' + '-' * 32)
        if args.model == 'CNN':
            classifier = ModelWiseCNNClassifier(class_num=backend_model_info.en_class_num)
            ckpt_name = 'cnn_con_model.pt'
        elif args.model == 'SeqXGPT':
            classifier = SeqXGPTModel(embedding_size=768, seq_len=1024, num_layers=6, id2labels=id2label)
            classifier.to("cuda" if torch.cuda.is_available() else "cpu")
            ckpt_name = 'seqxgpt_con_model.pt'
        else:
            classifier = ModelWiseTransformerClassifier(class_num=backend_model_info.en_class_num)
            ckpt_name = 'transformer_con_model.pt'

        trainer = SupervisedTrainer(data, classifier, loss_criterion = 'ContrastiveLoss')
        trainer.train(ckpt_name=ckpt_name)

    """classify after contrastive"""
    if args.train_mode == 'contrastive_classify':
        print('-' * 32 + 'contrastive_classify' + '-' * 32)
        if args.model == 'CNN':
            classifier = ModelWiseCNNClassifier(class_num=backend_model_info.en_class_num)
            ckpt_name = 'cnn_cc_model.pt'
            saved_model = torch.load(ckpt_name, weights_only=True)
            classifier.load_state_dict(saved_model.state_dict())
        elif args.model == 'SeqXGPT':
            classifier = SeqXGPTModel(embedding_size=768, seq_len=1024, num_layers=6, id2labels=id2label)
            classifier.to("cuda" if torch.cuda.is_available() else "cpu")
            ckpt_name = 'seqxgpt_cc_model.pt'
            saved_model = torch.load(ckpt_name, weights_only=True)
            classifier.load_state_dict(saved_model.state_dict())
        else:
            classifier = ModelWiseTransformerClassifier(class_num=backend_model_info.en_class_num)
            ckpt_name = 'transformer_cc_model.pt'
            saved_model = torch.load(ckpt_name, weights_only=True)
            classifier.load_state_dict(saved_model.state_dict())

        # trainer = SupervisedTrainer(data, classifier, train_mode='Contrastive_Classifier')
        trainer = SupervisedTrainer(data, classifier)
        trainer.train(ckpt_name=ckpt_name)
