# Transformers Pre-Training: Masked Language Modeling

Masked Language Modeling (MLM) is a pre-training technique designed to help transformers understand language semantics by predicting missing words in a sentence.

The pre-trained model can then be fine-tuned for downstream tasks such as classification, text generation, and more.

## Example Prediction
```py
>>> predict_mask('hello! how are you?')
masked: you predicted: you
ACTUAL: hello! how are you?
MASKED: hello! how are [MASK]?                                       
MODEL: hello! how are you?
```

## Objectives
✅ Implementing an Encoder-Only Transformer model  
✅ Preparing the dataset from scratch  
✅ Training a BERT-like tokenizer from scratch  
✅ Training from scratch with the MLM objective  
✅ Trained on a Wikipedia dataset from scratch  

---

## Preparing the Dataset
- In Masked Language Modeling, the loss calculation follows a similar approach to causal language models.
- The inputs and the labels are identical in terms of position.
- In the inputs:
    - 15% of the tokens are randomly masked using the `[MASK]` token.
    - Padding tokens are excluded from this 15%.
- In the labels:
    - Ground-truth tokens that were masked in the inputs remain visible.
    - All other tokens are ignored (set to `-100`), as per the default behavior of `nn.CrossEntropyLoss`.

### Sample Format
- `1` represents the `[PAD]` token.
- `2` represents the `[MASK]` token.
- Maximum sequence length = 128.

```py
tensor([5680,   10,  313,    2,    2, 4541,   14, 5393, 5404,   70,   11,  153,
          40, 2319,    2, 7560,   14, 1681, 3534,  148, 1649,   16,    1,    1,
           ...
        ])
tensor([-100, -100, -100,  939, 1058, -100, -100, -100, -100, -100, -100, -100,
        -100, -100, 1246, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        ...
        ])
```

---

## Encoder-Only Transformer Model
- Simple and efficient implementation.
- `nn.LayerNorm` replaced with `RMSNorm`, which is often preferred.
- Similar to BERT but with a simplified structure.
- No `[CLS]` or `[SEP]` tokens since only the MLM part is implemented.
- Uses learned positional embeddings instead of sinusoidal embeddings in BERT.
- Attention masks ensure that padding tokens are ignored.
- Currently supports batch size of `1` for inference.

### Inference Process
1. The encoder outputs pass through a `dim -> vocab` linear layer.
2. The logits at the masked token positions are extracted.
3. The softmax function is applied, and `argmax` is used to predict the missing token.

```
out: 1 x 128 x 256
if the input sequence for inference was masked at position 4, we extract 1 x 256 at index 4:
preds: out[:,4,:]
softmax -> argmax
preds: predicted token
```

---

## Model Performance Examples
> While the predicted word may not be exact, it captures the context of the sentence effectively.

```
masked: feed predicted: feed
ACTUAL: The larvae are black and flattened and feed on snails as well.
MASKED: the larvae are black and flattened and [MASK] on snails as well.                                
MODEL: the larvae are black and flattened and feed on snails as well.

---

masked: facility predicted: school
ACTUAL: Throughout the year, the facility hosts a variety of educational programs.
MASKED: throughout the year, the [MASK] hosts a variety of educational programs.
MODEL: throughout the year, the school hosts a variety of educational programs.

---

masked: provide predicted: create
ACTUAL: IRIS has partnered with other NGOs to provide funding for services, such as with The Vision Charity in Sri Lanka.
MASKED: iris has partnered with other ngos to [MASK] funding for services, such as with the vision charity in sri lanka.                       
MODEL: iris has partnered with other ngos to create funding for services, such as with the vision charity in sri lanka.
```

For more examples, check the notebook.

---

## Conclusion
### Thank You for Checking Out This Project! 🎉

I appreciate your time in exploring this project. I hope it has been helpful and insightful! Feel free to experiment, tweak, and enhance it as learning never stops. 😊

💡 Follow me for more:
- 🔗 [LinkedIn](http://www.linkedin.com/in/ahmed-nazeh10)
- 🔗 [GitHub](https://github.com/AhmedNazeh2)

💬 Share your thoughts and suggestions for improvement!
👍 If you found this helpful, don’t forget to give it a **star** on GitHub!

## Thank you! 🚀

