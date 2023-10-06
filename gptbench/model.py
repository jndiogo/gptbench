"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math, pdb

import torch
import torch.nn as nn
from torch.nn import functional as F

from .conf import Conf
from .utils import top_p


# -----------------------------------------------------------------------------

class GPT(nn.Module):
    """ GPT Language Model """

    @staticmethod
    def get_default_config():
        c = Conf()

        # device to run on
        c.setup('device', 'auto', str, "Device for running the net: 'cuda', 'cpu' or any other supported by Torch")

        c.setup('dtype', 'float32', str, "Type: 'float32', 'bfloat16'")

        c.setup('n_layer', None, int, 'Number of transformer blocks')
        c.setup('n_head', None, int, 'Number of attention heads')
        c.setup('n_embd', None, int, 'Number of embedding dimensions. Must be a multiple of n_head')
        
        # these options must be filled in externally
        c.setup('vocab_size', None, int, 'Size of the vocabulary. Must be set from dataset in use')
        c.setup('block_size', None, int, 'Block size: number of vocabulary items processed. Must be set')
        
        # dropout hyperparameter
        c.setup('dropout', 0.1, float, 'Dropout hyperparameter')

        return c
        


    @classmethod
    def from_pretrained(cls, model_type, model_config_override=None):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        # create a from-scratch initialized minGPT model
        config = cls.get_default_config()
        config.vocab_size = 50257 # openai's model vocabulary
        config.block_size = 1024  # openai's model block_size

        if model_config_override is not None:
            config.update(model_config_override, accept_keys=["dropout", "device", "dtype"])

        model = GPT(config, model_type)

        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param


        # init a huggingface/transformers model
        from transformers import GPT2LMHeadModel
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore: buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla nn.Linear.
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys) == len(sd_keys_hf), f"mismatched keys: {len(sd_keys)} != {len(sd_keys_hf)}"

        for k in sd_keys_hf:

            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model, config




    def __init__(self, config, model_type=None):

        """ model_type: a predefined model configuration: 'gpt-2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl' """

        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        self.block_size = config.block_size
        self.vocab_size = config.vocab_size

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device

        # 'float16': torch.float16 not used to avoid complications with smaller exponent range
        self.dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[config.dtype]


        type_given = model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given, "Only one of model_type or config.n_layer/n_head/config.n_embd must be given" # exactly one of these (XOR)

        if type_given:
            # translate from model_type to detailed configuration
            config.update({
                # names follow the huggingface naming conventions
                # GPT-1
                #'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # GPT-2 configs
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
                # Gophers
                #'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # I made these tiny models up
                #'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                #'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                #'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[model_type])

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.vocab_size, config.n_embd),
            wpe = nn.Embedding(self.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, self.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        self.n_params = sum(p.numel() for p in self.transformer.parameters())

        self.to(device=self.device, dtype=self.dtype)



    def get_num_params(self):
        return self.n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)



    def configure_optimizers(self, trainer_config):

        if trainer_config.optimizer == 'sgd': # SGD
            optimizer = torch.optim.SGD(self.parameters(), lr=trainer_config.learning_rate)

        elif trainer_config.optimizer == 'adamw': # AdamW
            """
            This long function is unfortunately doing something very simple and is being very defensive:
            We are separating out all parameters of the model into two buckets: those that will experience
            weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
            We are then returning the PyTorch optimizer object.
            """

            # separate out all parameters to those that will and won't experience regularizing weight decay
            decay = set()
            no_decay = set()
            whitelist_weight_modules = (torch.nn.Linear, )
            blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
            for mn, m in self.named_modules():
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                    # random note: because named_modules and named_parameters are recursive
                    # we will see the same tensors p many many times. but doing it this way
                    # allows us to know which parent module any tensor p belongs to...
                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)

            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in self.named_parameters()}
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert len(inter_params) == 0, "Parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
            assert len(param_dict.keys() - union_params) == 0, "Parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params), )

            # create the pytorch optimizer object
            optim_groups = [
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": trainer_config.adamw_weight_decay},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            ]
            optimizer = torch.optim.AdamW(optim_groups, lr=trainer_config.learning_rate, betas=(trainer_config.adamw_beta1,trainer_config.adamw_beta2))
        else:
            assert False, f"Unknown optimizer trainer.optimizer={trainer_config.optimizer}"

        return optimizer



    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

        

    @torch.no_grad()
    def generate(self, idx, 
                 max_new_tokens, temperature=1.0, do_sample=False, top=0, 
                 token_callback=None, stop_asap=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.

        top: 0: no top_k nor top_p
          ]0..1[: top_p(n)
          [-1..0[: top_k(vocab_size * -top)
          >=1: top_k(int(top))

        stop_asap=[False] - when set to True, must break and return

        """

        assert top <= self.vocab_size, f'Param top only up to vocab_size: {self.vocab_size}'

        assert len(idx), 'At least a single sample is needed to generate'

        if top >= -1. and top < 0.:
            top = max(1, int(self.vocab_size * -top))

        for i in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]

            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            if top >= 1.: # top_k: optionally crop the logits to only the top k options
                v, _ = torch.topk(logits, k=int(top))
                logits[logits < v[:, [-1]]] = -float('Inf')

            elif top > 0: # top_p: keep only above cummulative probability
                logits = top_p(logits, top)

            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else: # take best prob
                _, idx_next = torch.topk(probs, k=1, dim=-1)


            stop=0
            if token_callback is not None:
                stop = token_callback(idx_next, islast=not bool(max_new_tokens-1-i)) or 0

            if stop == -1: # don't add and break
                break

            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

            if stop == 1: # added then break
                break

            if stop_asap is not None and stop_asap[0]:
                break

        return idx










# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.dropout), # residual dropout
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


