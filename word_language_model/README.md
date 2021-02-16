# Word-level language modeling FNN

This example trains a one layer FNN on a language modeling task.
By default, the training script uses the Wikitext-2 dataset, provided.
The trained model can then be used by the generate script to generate new text.

```bash 
python main.py  --cuda --epochs 7 --batch_size 128 --log-interval 2000 --n 11 --emsize 500 --nhid 500 --lr 1e-4 --tied # Train a FNN on Wikitext-2 with CUDA
python generate.py --n 11 --checkpoint model_fnn_best.pt --cuda  # Generate samples from the trained FNN model.
```
During training, if a keyboard interrupt (Ctrl-C) is received,
training is stopped and the current model is evaluated against the test dataset.

The `main.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help            show this help message and exit
  --data DATA           location of the data corpus
  --emsize EMSIZE       size of word embeddings
  --nhid NHID           number of hidden units per layer
  --nlayers NLAYERS     number of layers
  --lr LR               initial learning rate
  --clip CLIP           gradient clipping
  --epochs EPOCHS       upper epoch limit
  --batch_size N        batch size
  --bptt BPTT           sequence length
  --dropout DROPOUT     dropout applied to layers (0 = no dropout)
  --tied                tie the word embedding and softmax weights
  --seed SEED           random seed
  --cuda                use CUDA
  --log-interval N      report interval
  --save SAVE           path to save the final model
  --onnx-export ONNX_EXPORT
                        path to export the final model in onnx format
  --nhead NHEAD         the number of heads in the encoder/decoder of the
                        transformer model
```
Sample outputs of training:
```bash
| epoch   1 |  2000/16317 batches | lr 0.00 | ms/batch  7.59 | loss  7.34 | ppl  1544.96
| epoch   1 |  4000/16317 batches | lr 0.00 | ms/batch  7.33 | loss  6.65 | ppl   772.25
| epoch   1 |  6000/16317 batches | lr 0.00 | ms/batch  7.31 | loss  6.47 | ppl   647.87
| epoch   1 |  8000/16317 batches | lr 0.00 | ms/batch  7.31 | loss  6.35 | ppl   574.90
| epoch   1 | 10000/16317 batches | lr 0.00 | ms/batch  7.32 | loss  6.27 | ppl   530.93
| epoch   1 | 12000/16317 batches | lr 0.00 | ms/batch  7.33 | loss  6.20 | ppl   494.95
| epoch   1 | 14000/16317 batches | lr 0.00 | ms/batch  7.36 | loss  6.11 | ppl   449.27
| epoch   1 | 16000/16317 batches | lr 0.00 | ms/batch  7.33 | loss  6.05 | ppl   425.08
-----------------------------------------------------------------------------------------
| end of epoch   1 | time: 129.63s | valid loss  5.75 | valid ppl   314.34
......
====================================================================================
| End of training | test loss  5.24 | test ppl   189.51
====================================================================================
SpearmanrResult(correlation=0.32094756497766097, pvalue=1.0573681235152218e-05)
```

The generated texts from tied and non-tied model are saved respectively at generated_tied.txt and generated_non_tied.txt