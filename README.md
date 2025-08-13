# CS366

## assignment1 Building a Transformer LM

### Tokenizer

* 初步实现了分词器的训练、解码以及编码
* 通过了BPE的训练测试以及tokenizer的测试

### Transformer LM

### Cross-entropy loss and AdamW optimizer

### Training loop

## 使用例
```
uv run pytest -q tests/test_train_bpe.py
```
* -q 快速核对
* -vv 详细查报错