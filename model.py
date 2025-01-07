# import torch
# import torch.nn as nn
# from transformers import AutoTokenizer, AutoModel
# import esm

# # Model Architecture
# class GRNAGenerator(nn.Module):
#     def __init__(self, esm_model_name="esm2_t6_8M_UR50D", freeze_encoder=True):
#         super().__init__()
        
#         # Encoder: Pre-trained ESM-2 (frozen) + bidirectional transformer
#         self.esm_encoder = AutoModel.from_pretrained(esm_model_name)
#         if freeze_encoder:
#             for param in self.esm_encoder.parameters():
#                 param.requires_grad = False
                
#         self.encoder_transformer = nn.TransformerEncoderLayer(
#             d_model=320,  # ESM-2 8M output dimension
#             nhead=8,
#             batch_first=True,
#             bidirectional=True
#         )
        
#         # Decoder: 3 autoregressive transformer layers
#         self.decoder = nn.TransformerDecoder(
#             decoder_layer=nn.TransformerDecoderLayer(
#                 d_model=320,
#                 nhead=8,
#                 batch_first=True
#             ),
#             num_layers=3
#         )
        
#         # Output projection
#         self.output_projection = nn.Linear(320, RNA_VOCAB_SIZE)
        
#     def forward(self, protein_ids, rna_ids=None):
#         # Encode protein sequence
#         encoder_output = self.esm_encoder(protein_ids).last_hidden_state
#         encoder_output = self.encoder_transformer(encoder_output)
        
#         # Decode RNA sequence
#         if self.training:
#             # Training mode: use teacher forcing
#             decoder_output = self.decoder(
#                 tgt=self.embedding(rna_ids),
#                 memory=encoder_output,
#                 tgt_mask=self.generate_square_subsequent_mask(rna_ids.size(1))
#             )
#         else:
#             # Inference mode: generate autoregressively
#             decoder_output = self.generate(encoder_output)
            
#         return self.output_projection(decoder_output)
    
#     def generate_square_subsequent_mask(self, sz):
#         mask = torch.triu(torch.ones(sz, sz), diagonal=1)
#         mask = mask.masked_fill(mask==1, float('-inf'))
#         return mask


import esm
import torch.nn as nn


class GuideRNAModel(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_dim, num_decoder_layers, vocab_size):
        super(GuideRNAModel, self).__init__()

        # Load pre-trained ESM-2 model from the esm package
        # esm.pre
        # model, alphabet = pretrained.ESM3_structure_encoder_v0()
        # esm.pretrained.esm2_t6_8M_UR50D()
        self.model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()  # Load the model

        # Tokenizer associated with the model (alphabet maps sequences to token IDs)
        self.batch_converter = alphabet.get_batch_converter()

        # Freeze ESM-2 weights
        for param in self.model.parameters():
            param.requires_grad = False

        # Bidirectional Transformer Layer (for context embedding)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=ff_dim)
        
        # Decoder with autoregressive layers
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=ff_dim)
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection layer (to generate the sequence)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, data, src_mask=None, tgt_mask=None, memory_mask=None):
        # Encoder: Use ESM-2 to encode the input sequences (protein or RNA)
        # ESM-2 expects input as a tensor of integers (tokens)
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        encoder_output = self.model(batch_tokens)[0]
        
        # Transformer encoder for further context embedding
        encoder_output = self.transformer_layer(encoder_output)
        
        # Decoder: Autoregressive decoding with cross-attention to encoder
        decoder_output = tgt
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, tgt_mask=tgt_mask, memory_mask=memory_mask)
        
        # Output layer to predict next RNA sequence
        output = self.output_layer(decoder_output)
        
        return output