import numpy as np
import os
import random
import torch
import json
import pandas as pd
import pickle

from tqdm import tqdm
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from torch.utils.data.dataloader import DataLoader, RandomSampler, SequentialSampler
from sklearn.preprocessing import normalize


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class DataManager:

    def __init__(self, train_path, test_path, batch_size, max_len, human_label, id2label, word_pad_idx=0, label_pad_idx=-1):
        set_seed(0)
        self.batch_size = batch_size
        self.max_len = max_len
        self.human_label = human_label
        # id2label: Mapping from IDs to labels.id2label.
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}
        # word_pad_idx, label_pad_idx: Padding indices for words and labels.
        self.word_pad_idx = word_pad_idx
        self.label_pad_idx = label_pad_idx

        data = dict()

        if train_path:
            # Process the training dataset and convert to Hugging Face Dataset format.
            # {'features': [], 'prompt_len': [], 'label_int': [], 'text': []}
            train_dict = self.initialize_dataset(train_path)
            data["train"] = Dataset.from_dict(train_dict)
        
        if test_path:
            # Process the test dataset and convert to Hugging Face Dataset format.
            test_dict = self.initialize_dataset(test_path)
            data["test"] = Dataset.from_dict(test_dict)
        
        # Wrap processed data into a DatasetDict for easier management.
        datasets = DatasetDict(data)
        # Initialize training and test dataloaders.
        if train_path:
            self.train_dataloader = self.get_train_dataloader(datasets["train"])
        if test_path:
            self.test_dataloader = self.get_eval_dataloader(datasets["test"])
        
        ##############################
        # Check for overlap between train and test data
        train_texts = set([sample['text'] for sample in self.train_dataloader.dataset])
        test_texts = set([sample['text'] for sample in self.test_dataloader.dataset])

        overlap = train_texts.intersection(test_texts)
        print('total train samples:', len(train_texts))
        print(f"Overlap between train and test: {len(overlap)} samples")        
        ##############################
        
    def initialize_dataset(self, data_path, save_dir=''):
        # Generate a filename for saving processed data (currently unused).
        processed_data_filename = Path(data_path).stem + "_processed.pkl"
        processed_data_path = os.path.join(save_dir, processed_data_filename)

        # Load raw data based on file type (JSON or JSONL).
        with open(data_path, 'r', encoding='utf-8') as f:
            if data_path.endswith('json'):
                samples = json.load(f)
            else:
                samples = [json.loads(line) for line in f]

        # Initialize an empty dictionary to store processed samples.
        samples_dict = {
            'features': [],
            'prompt_len': [],
            'label': [],
            # 'label_int': [],
            'text': []
        }

        # Iterate over samples with a progress bar.
        for item in tqdm(samples):
            text = item['text']
            label = item['label']
            prompt_len = item.get('prompt_len', 0)
            # prompt_len = len(text)

            # Extract and align token-level features.
            label_int = item['label_int']
            # List of starting indices.
            begin_idx_list = item['begin_idx_list']
            # Maximum start index for alignment.
            ll_tokens_list = item['ll_tokens_list']

            begin_idx_list = np.array(begin_idx_list)
            # Get the maximum value in begin_idx_list, which indicates where we need to truncate.

            max_begin_idx = np.max(begin_idx_list)
            # Truncate token features based on `max_begin_idx`.
            for idx, ll_tokens in enumerate(ll_tokens_list):
                ll_tokens_list[idx] = ll_tokens[max_begin_idx:]

            # Align all token features to the shortest sequence length.
            min_len = np.min([len(ll_tokens) for ll_tokens in ll_tokens_list])

            # Align the lengths of all vectors
            for idx, ll_tokens in enumerate(ll_tokens_list):
                ll_tokens_list[idx] = ll_tokens[:min_len]
            
            # Skip invalid samples with no features.
            if len(ll_tokens_list) == 0 or len(ll_tokens_list[0]) == 0:
                continue

            # Transpose the token features for compatibility with the model.
            ll_tokens_list = np.array(ll_tokens_list)
            ll_tokens_list = ll_tokens_list.transpose()
            ll_tokens_list = ll_tokens_list.tolist()

            samples_dict['features'].append(ll_tokens_list)
            samples_dict['prompt_len'].append(prompt_len)
            samples_dict['label'].append(label)
            # samples_dict['label_int'].append(label_int)
            samples_dict['text'].append(text)

        return samples_dict

    def get_train_dataloader(self, dataset):
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          sampler=RandomSampler(dataset),
                          collate_fn=self.data_collator)

    def get_eval_dataloader(self, dataset):
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          sampler=SequentialSampler(dataset),
                          collate_fn=self.data_collator)
    
    def data_collator(self, samples):
        batch = {}

        # Extract fields from the samples.
        features = [sample['features'] for sample in samples]
        prompt_len = [sample['prompt_len'] for sample in samples]
        text = [sample['text'] for sample in samples]
        label = [sample['label'] for sample in samples]

        # Pad features and create masks.
        features, masks = self.process_and_convert_to_tensor(features)
        # pad_masks = ~masks * -1
        pad_masks = (1 - masks) * self.label_pad_idx

        # Align and pad labels for each sequence.
        for idx, p_len in enumerate(prompt_len):
            prefix_len = len(self.split_sentence(text[idx][:p_len]))
            # If the prompt exceeds the maximum length, truncate it.
            if prefix_len > self.max_len:
                prefix_ids = self.sequence_labels_to_ids(self.max_len, self.human_label)
                masks[idx][:] = prefix_ids[:]
                continue
            total_len = len(self.split_sentence(text[idx]))
            
            # Assign human labels to the prompt section.
            if prefix_len > 0:
                prefix_ids = self.sequence_labels_to_ids(prefix_len, self.human_label)
                masks[idx][:prefix_len] = prefix_ids[:]
            if total_len - prefix_len > 0:
                if total_len > self.max_len:
                    human_ids = self.sequence_labels_to_ids(self.max_len - prefix_len, label[idx])
                else:
                    human_ids = self.sequence_labels_to_ids(total_len - prefix_len, label[idx])
                masks[idx][prefix_len:total_len] = human_ids[:]
            masks[idx] += pad_masks[idx]

        # Construct the final batch dictionary.
        batch['features'] = features
        batch['labels'] = masks
        batch['text'] = text

        return batch

    def sequence_labels_to_ids(self, seq_len, label):
        """
        Converts sequence labels into IDs using prefix conventions:
        - B-: Beginning of a sequence.
        - M-: Middle of a sequence.
        - E-: End of a sequence.
        - S-: Single-token sequences.
        """
        if seq_len <= 0:
            return None
        return torch.tensor([self.label2id[label]] * seq_len, dtype=torch.long)
        # prefix = ['B-', 'M-', 'E-', 'S-']
        # if seq_len <= 0:
        #     return None
        # # Special case for single-token sequences.
        # elif seq_len == 1:
        #     # Assign the 'S-' prefix.
        #     label = 'S-' + label
        #     return torch.tensor([self.label2id[label]], dtype=torch.long)
        # # For sequences with more than one token:
        # else:
        #     ids = []
        #     # Add the 'B-' label for the start of the sequence.
        #     ids.append(self.label2id['B-'+label])
        #     # Add 'M-' labels for the middle tokens.
        #     ids.extend([self.label2id['M-'+label]] * (seq_len - 2))
        #     # Add the 'E-' label for the end of the sequence.
        #     ids.append(self.label2id['E-'+label])
        #     return torch.tensor(ids, dtype=torch.long)

    def process_and_convert_to_tensor(self, data):
        """ here, data is features. """
        max_len = self.max_len
        # Determine the dimensionality of each feature vector.
        # data shape: [B, S, E]
        feat_dim = len(data[0][0])
        # Pad sequences to `max_len` by appending zero vectors.
        padded_data = [  # [[0] * feat_dim] + 
            seq + [[0] * feat_dim] * (max_len - len(seq)) for seq in data
        ]
        # Truncate sequences to `max_len` if they exceed it.
        padded_data = [seq[:max_len] for seq in padded_data]

        # masks = [[False] * min(len(seq)+1, max_len) + [True] * (max_len - min(len(seq)+1, max_len)) for seq in data]
        masks = [[1] * min(len(seq), max_len) + [0] *
                (max_len - min(len(seq), max_len)) for seq in data]

        # Convert padded data and masks into PyTorch tensors.
        tensor_data = torch.tensor(padded_data, dtype=torch.float)
        tensor_mask = torch.tensor(masks, dtype=torch.long)

        return tensor_data, tensor_mask

    def _split_en_sentence(self, sentence, use_sp=False):
        import re
        # Match sequences of non-space characters or single spaces.
        pattern = re.compile(r'\S+|\s')
        # Tokenize the sentence into words and spaces.
        words = pattern.findall(sentence)
        # If subword processing is enabled:
        if use_sp:
            # Replace spaces with the subword token.
            words = ["▁" if item == " " else item for item in words]
        return words

    def _split_cn_sentence(self, sentence, use_sp=False):
        words = list(sentence)
        # If subword processing is enabled:
        if use_sp:
            # Replace spaces with the subword token.
            words = ["▁" if item == " " else item for item in words]
        return words

    # Splits sentences into tokens based on language characteristics.
    def split_sentence(self, sentence, use_sp=False, cn_percent=0.2):
        # Calculate the total number of characters in the sentence.
        total_char_count = len(sentence)
        
        # Avoid division by zero by incrementing total_char_count if it is zero.
        total_char_count += 1 if total_char_count == 0 else 0  
        
        # Count the number of Chinese characters in the sentence.
        chinese_char_count = sum('\u4e00' <= char <= '\u9fff' for char in sentence)
        
        # Determine whether the sentence is primarily Chinese.
        if chinese_char_count / total_char_count > cn_percent:
            # Use Chinese tokenization if the ratio of Chinese characters exceeds the threshold.
            return self._split_cn_sentence(sentence, use_sp)
        else:
            # Use English tokenization otherwise.
            return self._split_en_sentence(sentence, use_sp)
