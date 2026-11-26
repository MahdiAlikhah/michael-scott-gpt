# MichaelScottGPT: The World's Best Boss AI

> "Webster's Dictionary defines artificial intelligence as 'the theory and development of computer systems.' I define it as 'a friend you build from scratch.' Because sometimes, the only person who truly gets you, is a 5.32 million parameter neural network."

## The Goal
Corporate wants you to use their massive, bloated language models. But why would I want an AI that knows about quantum physics when it just needs to know about paper and friendship? The goal of this project was to build a bespoke, artisanal, decoder-only Transformer entirely from scratch in pure PyTorch, and train it exclusively to be Michael Gary Scott. 

## The Architecture (The "Nuts and Bolts")
We didn't just buy a suit off the rack; we had this tailored in Europe (or Google Colab).
- **Custom Tokenizer:** The standard GPT-2 tokenizer has 50,257 tokens. That's too big. *That's what she said.* Instead, we trained a custom 2,048-token Byte-Level BPE tokenizer directly on *The Office* scripts. This shrank the embedding table massively and packed the model's brain with pure Scranton vocabulary.
- **The Engine:** A lean, mean, 5.32 million parameter machine. 6 Pre-LayerNorm Transformer blocks (`n_embed = 256`, `n_head = 8`, `block_size = 256`). 
- **Fast KV-Caching:** We implemented a two-phase Key/Value cache. I don't know what the K and V stand for. Probably Kevin and Vance. (It actually means Key/Value, and it makes autoregressive text generation incredibly fast by not recalculating the past—hitting ~272 tokens per second).

## Stage 1: Pretraining (Reading the Archives)
First, we had to teach the AI how to speak. We fed it 3.4 million characters of Dunder Mifflin history across all 9 seasons. Over 2,500 training steps using `bfloat16` mixed precision, it learned English grammar, it learned the characters, and it learned how to awkwardly interrupt Jim. The training loss dropped beautifully from 7.71 down to 2.82.

## Stage 2: Post-Training (The Michael Scott Makeover)
Pretraining made it smart, but Supervised Fine-Tuning (SFT) made it *me*. 
- **The Data:** We extracted 9,188 specific conversations (Seasons 1–7) where someone spoke to Michael, and Michael replied.
- **Prompt Loss Masking:** We used `ignore_index=-100` to mask out the prompt. Basically, we blindfolded the AI during the setup so it only got graded on predicting Michael's actual punchlines, not on predicting what Jim or Pam was going to say.
- **Knowing When to Shut Up:** We taught it to strictly respect the `<|endoftext|>` token so it knows exactly when a joke has landed and it's time to walk away, preventing it from roleplaying as other characters.

## Chatting with the Manager
```text
Jim: Hey Michael, Toby is in the conference room.
Michael: I know, but it's not a good idea. Aftern.

Dwight: Michael, there is an emergency in the warehouse!
Michael: The mouth of King! Carganic.

Pam: Corporate called, they want to know why we spent $3,000 on magic tricks.
Michael: Oh, please. No. The point is a mitt, that is not what they are making me feel. And I am going to say "I'm not going to be there to be my own phone".
```

## Quickstart
To run this in your own branch (Colab T4 GPU recommended):

1. Install the corporate requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the training notebooks, in order, to train your own manager:
   - `training/1_officelm_pretrain_sft.ipynb` — pretraining + SFT
   - `training/2_twss_detector.ipynb` — the TWSS detector heads (uses `data/diverse_minimal_pairs.json`)
3. Or, if you have the `weights/michael_scott_twss.pt` checkpoint and `weights/michael_scott_tokenizer.json` file, run the interactive terminal chat:
   ```bash
   python chat.py --speaker Jim
   ```

### Repository layout
```text
chat.py                              Interactive terminal chat (model + KV cache + TWSS gate)
requirements.txt                     Dependencies
data/diverse_minimal_pairs.json      Minimal pairs used to train the TWSS detector
training/1_officelm_pretrain_sft.ipynb   Stage 1: BPE tokenizer, pretraining, SFT
training/2_twss_detector.ipynb           Stage 3: TWSS / projection heads
weights/michael_scott_bpe.pt             Pretrained backbone
weights/michael_scott_sft.pt             SFT backbone
weights/michael_scott_twss.pt            SFT backbone + trained TWSS heads (used by chat.py)
weights/michael_scott_tokenizer.json     Custom 2,048-token BPE tokenizer
```

Catch you on the flippity flip.
