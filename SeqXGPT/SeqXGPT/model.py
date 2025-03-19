import torch
import torch.nn as nn
from typing import List, Tuple
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers.models.bert import BertModel
from fastNLP.modules.torch import MLP,ConditionalRandomField,allowed_transitions
from torch.nn import CrossEntropyLoss
import random
import math
import torch.nn.functional as F


# Add this utility function at the top of the file, after imports
def get_positional_encoding(seq_length, embedding_size, device=None):
    """
    Generates sinusoidal positional encodings for transformer models.
    
    Args:
        seq_length: Maximum sequence length to generate positions for
        embedding_size: Dimension of the embeddings
        device: Device to create the tensor on (defaults to None)
        
    Returns:
        Tensor of shape (1, seq_length, embedding_size) containing positional encodings
    """
    pe = torch.zeros(seq_length, embedding_size, device=device)
    position = torch.arange(0, seq_length, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embedding_size, 2, dtype=torch.float, device=device) *
                         -(math.log(10000.0) / embedding_size))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # Add batch dimension (1, seq_length, embedding_size)


class SeqXGPTModel(nn.Module):
    def __init__(self, id2labels, embedding_size=128, num_heads=4,
                 intermediate_size=64, num_layers=2, dropout_rate=0.1):
        super().__init__()
        self.embedding_size = embedding_size

        # Transformer Encoder setup
        self.encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout_rate,
            batch_first=True
        )
        self.encoder = TransformerEncoder(encoder_layer=self.encoder_layer,
                                          num_layers=num_layers)
        
        # Normalization, classifier, dropout, and CRF
        self.norm = nn.LayerNorm(embedding_size)
        self.label_num = len(id2labels)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(nn.Linear(embedding_size, self.label_num))
        # self.crf = ConditionalRandomField(num_tags=self.label_num,
        #                                    allowed_transitions=allowed_transitions(id2labels))
        self.crf = ConditionalRandomField(num_tags=self.label_num, allowed_transitions=None)
        self.crf.trans_m.data *= 0

    def forward(self, inputs, labels, method="patch_average", patch_size=3,
                kernel_size=3, stride=1):
        """
        Unified forward method to match other models' interface
        inputs: Tensor of shape (batch, original_seq_len, embedding_size)
        labels: Tensor of shape (batch, original_seq_len) with -1 indicating padding
        """
        # Create the original valid token mask from labels (True for valid tokens)
        orig_mask = labels.gt(-1)  # shape: (batch, original_seq_len)
        
        # Preprocess inputs and at the same time process the mask and labels
        if method == "patch_average":
            inputs = self.patch_average(inputs, patch_size)
            mask = self.patch_mask(orig_mask, patch_size)
            proc_labels = self.patch_labels(labels, patch_size)
        elif method == "convolution_like":
            inputs = self.convolution_like(inputs, kernel_size, stride)
            mask = self.convolution_like_mask(orig_mask, kernel_size, stride)
            proc_labels = self.convolution_like_labels(labels, kernel_size, stride)
        elif method == "patch_shuffle":
            inputs = self.patch_shuffle(inputs, patch_size)
            mask = self.patch_shuffle_mask(orig_mask, patch_size)
            proc_labels = self.patch_shuffle_labels(labels, patch_size)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        #=============================================#
        # Ensure mask has the same sequence length as inputs
        current_seq_length = inputs.size(1)
        if mask.size(1) != current_seq_length:
            # Either truncate or pad the mask to match the sequence length
            if mask.size(1) > current_seq_length:
                mask = mask[:, :current_seq_length]
            else:
                pad_size = current_seq_length - mask.size(1)
                mask = F.pad(mask, (0, pad_size), value=False)
            
            # Also adjust proc_labels to match
            if proc_labels.size(1) != current_seq_length:
                if proc_labels.size(1) > current_seq_length:
                    proc_labels = proc_labels[:, :current_seq_length]
                else:
                    pad_size = current_seq_length - proc_labels.size(1)
                    proc_labels = F.pad(proc_labels, (0, pad_size), value=-1)
        #=============================================#
        
        # Transformer expects a padding mask where True indicates a padded token.
        padding_mask = ~mask  # shape: (batch, new_seq_len)
        
        # Dynamically generate positional encoding for the new sequence length.
        # current_seq_length = inputs.size(1)
        pos_encoding = get_positional_encoding(current_seq_length, self.embedding_size, inputs.device)
                
        # Add positional encoding to the processed inputs.
        outputs = inputs + pos_encoding  # shape: (batch, new_seq_len, embedding_size)
        outputs = self.norm(outputs)
        outputs = self.encoder(outputs, src_key_padding_mask=padding_mask)
        dropout_outputs = self.dropout(outputs)
        logits = self.classifier(dropout_outputs)

        if self.training:
            # # Calculate class weights based on label distribution
            # label_counts = torch.bincount(proc_labels[proc_labels != -1])
            # weights = 1.0 / (label_counts.float() / label_counts.sum())
            # # Normalize weights
            # weights = weights / weights.sum() * len(weights)
            # loss_fct = nn.CrossEntropyLoss(weight=weights, ignore_index=-1)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.label_num), proc_labels.view(-1))
            return {
                    'loss': loss,
                    'logits': logits
                    }
        else:
            paths, scores = self.crf.viterbi_decode(logits=logits, mask=mask)
            paths[mask == 0] = -1
            return {
                    'preds': paths,
                    'logits': logits
                    }

    # -------------------------
    # Preprocessing Functions
    # -------------------------
    def patch_average(self, wave, patch_size):
        batch_size, seq_len, emb_size = wave.size()
        pad_size = (patch_size - (seq_len % patch_size)) % patch_size
        if pad_size > 0:
            wave = F.pad(wave, (0, 0, 0, pad_size))
        wave = wave.view(batch_size, -1, patch_size, emb_size)
        return wave.mean(dim=2)

    def convolution_like(self, wave, kernel_size, stride):
        """Improved convolution-like implementation with padding"""
        batch_size, seq_length, emb_size = wave.size()
        patches = []
        
        # Add padding to ensure all tokens are covered
        padding = (kernel_size - stride) // 2
        padded_wave = F.pad(wave, (0, 0, padding, padding), mode='constant', value=0)
        
        for i in range(0, seq_length + 2*padding - kernel_size + 1, stride):
            patch = padded_wave[:, i:i+kernel_size]
            patch_mean = torch.mean(patch, dim=1, keepdim=True)
            patches.append(patch_mean)
        
        return torch.cat(patches, dim=1)

    def patch_shuffle(self, wave, patch_size):
        """Splits the wave into patches, shuffles them, and concatenates along the time dimension."""
        batch_size, seq_length, emb_size = wave.size()
        # Calculate number of full patches
        num_patches = seq_length // patch_size
        if num_patches * patch_size < seq_length:
            num_patches += 1  # Handle residual tokens
        # Create padding if needed
        padded_length = num_patches * patch_size
        if padded_length > seq_length:
            padding = torch.zeros(batch_size, padded_length - seq_length, emb_size, 
                                device=wave.device)
            wave = torch.cat([wave, padding], dim=1)
        # Split into patches and shuffle
        patches = wave.view(batch_size, num_patches, patch_size, emb_size)
        # Create random permutation for each sample in batch
        shuffle_idx = torch.stack([torch.randperm(num_patches) for _ in range(batch_size)])
        # Shuffle using gathered indices
        patches = patches.gather(1, shuffle_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, patch_size, emb_size))
        # Reconstruct sequence
        return patches.view(batch_size, num_patches * patch_size, emb_size)

    # -------------------------
    # Functions for Processing the Mask and Labels
    # -------------------------
    def patch_mask(self, mask, patch_size):
        """
        Process the mask by patching.
        For each patch, if any token is valid, the entire patch is marked valid.
        Returns a tensor of shape (batch, num_patches).
        """
        patches = []
        seq_length = mask.size(1)
        for i in range(0, seq_length, patch_size):
            patch = mask[:, i:i+patch_size]  # shape: (batch, patch_size)
            patch_valid = patch.any(dim=1, keepdim=True)  # shape: (batch, 1)
            patches.append(patch_valid)
        return torch.cat(patches, dim=1)

    def convolution_like_mask(self, mask, kernel_size, stride):
        """Process the mask with a sliding window."""
        patches = []
        seq_length = mask.size(1)
        for i in range(0, seq_length - kernel_size + 1, stride):
            patch = mask[:, i:i+kernel_size]
            patch_valid = patch.any(dim=1, keepdim=True)
            patches.append(patch_valid)
        return torch.cat(patches, dim=1)

    def patch_shuffle_mask(self, mask, patch_size):
        """Splits the mask into patches, shuffles them, and concatenates back."""
        patches = [mask[:, i:i+patch_size] for i in range(0, mask.size(1), patch_size)]
        random.shuffle(patches)
        return torch.cat(patches, dim=1)

    # Modify patch_labels to better preserve label information
    def patch_labels(self, labels, patch_size):
        batch_size, seq_len = labels.shape
        pad_size = (patch_size - (seq_len % patch_size)) % patch_size
        padded = F.pad(labels, (0, pad_size), value=-1)
        # Instead of just taking first label, use majority voting within each patch
        patches = padded.view(batch_size, -1, patch_size)
        result = []
        for b in range(batch_size):
            batch_result = []
            for p in range(patches.size(1)):
                patch = patches[b, p]
                # Filter out padding
                valid_labels = patch[patch != -1]
                if len(valid_labels) > 0:
                    # Use most common label in patch
                    values, counts = torch.unique(valid_labels, return_counts=True)
                    batch_result.append(values[counts.argmax()].item())
                else:
                    batch_result.append(-1)
            result.append(batch_result)
        return torch.tensor(result, device=labels.device)

    def convolution_like_labels(self, labels, kernel_size, stride):
        """Processes labels with a sliding window."""
        patches = []
        seq_length = labels.size(1)
        for i in range(0, seq_length - kernel_size + 1, stride):
            patch = labels[:, i:i+kernel_size]
            patch_label = patch[:, 0:1]
            patches.append(patch_label)
        return torch.cat(patches, dim=1)

    def patch_shuffle_labels(self, labels, patch_size):
        """Splits the labels into patches, shuffles them, and concatenates."""
        patches = [labels[:, i:i+patch_size] for i in range(0, labels.size(1), patch_size)]
        random.shuffle(patches)
        return torch.cat(patches, dim=1)


# Feature Extraction: CNN.
# Contextual Understanding: CRF.
# Position Encoding: None.
class ConvFeatureExtractionModel(nn.Module):

    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        conv_dropout: float = 0.0,
        conv_bias: bool = False,
    ):
        super().__init__()

        def block(n_in, n_out, k, stride=1, conv_bias=False):
            padding = k // 2
            return nn.Sequential(
                nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=k, stride=stride, padding=padding, bias=conv_bias),
                nn.Dropout(conv_dropout),
                nn.ReLU(),
            )

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for _, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(in_d, dim, k, stride=stride, conv_bias=conv_bias))
            in_d = dim

    def forward(self, x):
        # x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        return x


# Convolution-based sequence labeling model with CRF.
class ModelWiseCNNClassifier(nn.Module):

    def __init__(self, id2labels, dropout_rate=0.1):
        super(ModelWiseCNNClassifier, self).__init__()
        feature_enc_layers = [(64, 5, 1)] + [(128, 3, 1)] * 3 + [(64, 3, 1)]
        self.conv = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            conv_dropout=0.0,
            conv_bias=False,
        )

        embedding_size = 4 *64
        self.norm = nn.LayerNorm(embedding_size)
        
        self.label_num = len(id2labels)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(nn.Linear(embedding_size, self.label_num))
        # Conditional Random Field (CRF) ensures structured predictions with label dependencies.
        # Allowed Transitions: Defined by id2labels for valid label sequences.
        # self.crf = ConditionalRandomField(num_tags=self.label_num, allowed_transitions=allowed_transitions(id2labels))
        self.crf = ConditionalRandomField(num_tags=self.label_num, allowed_transitions=None)
        self.crf.trans_m.data *= 0

    def conv_feat_extract(self, x):
        out = self.conv(x)
        out = out.transpose(1, 2)
        return out

    def forward(self, x, labels):
        x = x.transpose(1, 2)
        out1 = self.conv_feat_extract(x[:, 0:1, :])  
        out2 = self.conv_feat_extract(x[:, 1:2, :])  
        out3 = self.conv_feat_extract(x[:, 2:3, :])  
        out4 = self.conv_feat_extract(x[:, 3:4, :])  
        outputs = torch.cat((out1, out2, out3, out4), dim=2)  
        
        outputs = self.norm(outputs)
        dropout_outputs = self.dropout(outputs)
        logits = self.classifier(dropout_outputs)
        
        if self.training:
            # Training mode: Cross-entropy loss.
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.label_num), labels.view(-1))
            output = {'loss': loss, 'logits': logits}
        else:
            # Inference mode: Viterbi decoding with CRF.
            mask = labels.gt(-1)
            paths, scores = self.crf.viterbi_decode(logits=logits, mask=mask)
            paths[mask==0] = -1
            output = {'preds': paths, 'logits': logits}
            pass

        return output
    

# Feature Extraction: CNN + Transformer.
# Contextual Understanding: Transformer + CRF.
# Position Encoding: Sinusoidal Positional Encoding.
class ModelWiseTransformerClassifier(nn.Module):

    def __init__(self, id2labels, seq_len, intermediate_size = 512, num_layers=2, dropout_rate=0.1):
        super(ModelWiseTransformerClassifier, self).__init__()
        feature_enc_layers = [(64, 5, 1)] + [(128, 3, 1)] * 3 + [(64, 3, 1)]
        self.conv = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            conv_dropout=0.0,
            conv_bias=False,
        )
        
        self.seq_len = seq_len
        embedding_size = 4 *64
        self.encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=16,
            dim_feedforward=intermediate_size,
            dropout=dropout_rate,
            batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer=self.encoder_layer,
                                            num_layers=num_layers)
        self.register_buffer(
            'position_encoding', 
            get_positional_encoding(seq_len, embedding_size)
        )
        
        self.norm = nn.LayerNorm(embedding_size)
        
        self.label_num = len(id2labels)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(nn.Linear(embedding_size, self.label_num))
        self.crf = ConditionalRandomField(num_tags=self.label_num, allowed_transitions=allowed_transitions(id2labels))
        self.crf.trans_m.data *= 0

    def conv_feat_extract(self, x):
        out = self.conv(x)
        out = out.transpose(1, 2)
        return out

    def forward(self, x, labels):
        mask = labels.gt(-1)
        padding_mask = ~mask

        # Processes inputs with CNN, adds positional encoding,
        # and passes through the Transformer encoder.
        x = x.transpose(1, 2)
        out1 = self.conv_feat_extract(x[:, 0:1, :])  
        out2 = self.conv_feat_extract(x[:, 1:2, :])  
        out3 = self.conv_feat_extract(x[:, 2:3, :])  
        out4 = self.conv_feat_extract(x[:, 3:4, :])  
        out = torch.cat((out1, out2, out3, out4), dim=2)  
        
        outputs = out + self.position_encoding.to(out.device)
        outputs = self.norm(outputs)
        outputs = self.encoder(outputs, src_key_padding_mask=padding_mask)
        dropout_outputs = self.dropout(outputs)
        logits = self.classifier(dropout_outputs)
        
        if self.training:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.label_num), labels.view(-1))
            output = {'loss': loss, 'logits': logits}
        else:
            paths, scores = self.crf.viterbi_decode(logits=logits, mask=mask)
            paths[mask==0] = -1
            output = {'preds': paths, 'logits': logits}
            pass

        return output
    

# Feature Extraction: Transformer.
# Contextual Understanding: Transformer + CRF.
# Position Encoding: Sinusoidal Positional Encoding.
# Directly process sequences using a Transformer encoder.
class TransformerOnlyClassifier(nn.Module):

    def __init__(self, id2labels, seq_len, embedding_size=64, num_heads=2, intermediate_size=64, num_layers=2, dropout_rate=0.1):
        super(TransformerOnlyClassifier, self).__init__()

        self.encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout_rate,
            batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer=self.encoder_layer,
                                            num_layers=num_layers)

        self.position_encoding = torch.zeros((seq_len, embedding_size))
        for pos in range(seq_len):
            for i in range(0, embedding_size, 2):
                self.position_encoding[pos, i] = torch.sin(
                    torch.tensor(pos / (10000**((2 * i) / embedding_size))))
                self.position_encoding[pos, i + 1] = torch.cos(
                    torch.tensor(pos / (10000**((2 * (i + 1)) / embedding_size))))
        # self.register_buffer(
        #     'position_encoding', 
        #     get_positional_encoding(seq_len, embedding_size)
        # )
        
        self.norm = nn.LayerNorm(embedding_size)
        
        self.label_num = len(id2labels)
        self.dropout = nn.Dropout(dropout_rate)
        # self.classifier = nn.Sequential(nn.Linear(embedding_size, self.label_num))
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size, embedding_size // 2),
            nn.ReLU(),
            nn.Linear(embedding_size // 2, self.label_num)
        )
        # self.crf = ConditionalRandomField(num_tags=self.label_num,
        #                                   allowed_transitions=allowed_transitions(id2labels))
        self.crf = ConditionalRandomField(num_tags=self.label_num, allowed_transitions=None)
        self.crf.trans_m.data *= 0
    
    def forward(self, inputs, labels):
        mask = labels.gt(-1)
        padding_mask = ~mask
        
        outputs = inputs + self.position_encoding.to(inputs.device)
        outputs = self.norm(outputs)
        outputs = self.encoder(outputs, src_key_padding_mask=padding_mask)
        dropout_outputs = self.dropout(outputs)
        logits = self.classifier(dropout_outputs)
        
        if self.training:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.label_num), labels.view(-1))
            output = {'loss': loss, 'logits': logits}
        else:
            paths, scores = self.crf.viterbi_decode(logits=logits, mask=mask)
            paths[mask==0] = -1
            output = {'preds': paths, 'logits': logits}
            pass

        return output