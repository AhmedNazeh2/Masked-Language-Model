from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer
from tokenizers import decoders
from pathlib import Path

lines = [] # lines from the dataset, see notebook

bert_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
bert_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
bert_tokenizer.pre_tokenizer = Whitespace()
bert_tokenizer.decoder = decoders.WordPiece()

trainer = WordPieceTrainer(special_tokens=["[UNK]","[PAD]", "[MASK]"],vocab_size=8192)
bert_tokenizer.train_from_iterator(lines,trainer)
bert_tokenizer.enable_padding(pad_id=bert_tokenizer.token_to_id('[PAD]'),length=128)
bert_tokenizer.enable_truncation(128)

base = Path('mlm-baby-bert/tokenizer',)
base.mkdir(exist_ok=True,parents=True)
bert_tokenizer.save(str(base / 'wiki.json'))