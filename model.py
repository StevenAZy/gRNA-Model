import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import esm

# Model Architecture
class GRNAGenerator(nn.Module):
    def __init__(self, esm_model_name="esm2_t6_8M_UR50D", freeze_encoder=True):
        super().__init__()
        
        # Encoder: Pre-trained ESM-2 (frozen) + bidirectional transformer
        self.esm_encoder = AutoModel.from_pretrained(esm_model_name)
        if freeze_encoder:
            for param in self.esm_encoder.parameters():
                param.requires_grad = False
                
        self.encoder_transformer = nn.TransformerEncoderLayer(
            d_model=320,  # ESM-2 8M output dimension
            nhead=8,
            batch_first=True,
            bidirectional=True
        )
        
        # Decoder: 3 autoregressive transformer layers
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=320,
                nhead=8,
                batch_first=True
            ),
            num_layers=3
        )
        
        # Output projection
        self.output_projection = nn.Linear(320, RNA_VOCAB_SIZE)
        
    def forward(self, protein_ids, rna_ids=None):
        # Encode protein sequence
        encoder_output = self.esm_encoder(protein_ids).last_hidden_state
        encoder_output = self.encoder_transformer(encoder_output)
        
        # Decode RNA sequence
        if self.training:
            # Training mode: use teacher forcing
            decoder_output = self.decoder(
                tgt=self.embedding(rna_ids),
                memory=encoder_output,
                tgt_mask=self.generate_square_subsequent_mask(rna_ids.size(1))
            )
        else:
            # Inference mode: generate autoregressively
            decoder_output = self.generate(encoder_output)
            
        return self.output_projection(decoder_output)
    
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask