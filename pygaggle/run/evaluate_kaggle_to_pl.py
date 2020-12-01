from typing import Optional, List
from pathlib import Path
import logging

from pydantic import BaseModel, validator
from transformers import (AutoModel,
                          AutoModelForQuestionAnswering,
                          AutoModelForSequenceClassification,
                          AutoTokenizer,
                          BertForQuestionAnswering,
                          BertForSequenceClassification)
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from .args import ArgumentParserBuilder, opt
from pygaggle.rerank.base import Reranker
from pygaggle.rerank.bm25 import Bm25Reranker
from pygaggle.rerank.transformer import (
    QuestionAnsweringTransformerReranker,
    MonoBERT,
    MonoT5,
    UnsupervisedTransformerReranker
    )
from pygaggle.rerank.random import RandomReranker
from pygaggle.rerank.similarity import CosineSimilarityMatrixProvider
from pygaggle.model import (CachedT5ModelLoader,
                            RerankerEvaluator,
                            SimpleBatchTokenizer,
                            T5BatchTokenizer,
                            metric_names)
from pygaggle.data import LitReviewDataset
from pygaggle.settings import Cord19Settings

SETTINGS = Cord19Settings()
METHOD_CHOICES = ('transformer', 'bm25', 't5', 'seq_class_transformer',
                  'qa_transformer', 'random')

class KaggleEvaluationOptions(BaseModel):
    dataset: Path
    index_dir: Path
    method: str
    batch_size: int
    device: str
    split: str
    do_lower_case: bool
    metrics: List[str]
    model_name: Optional[str]
    tokenizer_name: Optional[str]

    @validator('dataset')
    def dataset_exists(cls, v: Path):
        assert v.exists(), 'dataset must exist'
        return v

    @validator('model_name')
    def model_name_sane(cls, v: Optional[str], values, **kwargs):
        method = values['method']
        if method == 'transformer' and v is None:
            raise ValueError('transformer name must be specified')
        elif method == 't5':
            return SETTINGS.t5_model_type
        if v == 'biobert':
            return 'monologg/biobert_v1.1_pubmed'
        return v

    @validator('tokenizer_name')
    def tokenizer_sane(cls, v: str, values, **kwargs):
        if v is None:
            return values['model_name']
        return v
    
    


    def construct_t5(options: KaggleEvaluationOptions) -> Reranker:
        model_loader = CachedT5ModelLoader(SETTINGS.t5_model_dir,
                                    SETTINGS.cache_dir,
                                    'ranker',
                                    SETTINGS.t5_model_type,
                                    SETTINGS.flush_cache)
        device = torch.device(options.device)
        model = model_loader.load().to(device).eval()
        tokenizer = MonoT5.get_tokenizer(options.model_type,
                                        do_lower_case=options.do_lower_case,
                                        batch_size=options.batch_size)
        return {'model_loader': model_loader, 'device' : device, 'model' : model, 'tokenizer' : tokenizer, 'reranker' : MonoT5(model, tokenizer)}

class MyIterableDataset(datasets):
    def __init__(self, options: KaggleEvaluationOptions, transform=None):
        """
        Args:
            options: Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ds = LitReviewDataset.from_file(str(options.dataset))
        self.split = options.split
        self.loader = Cord19DocumentLoader(str(options.index_dir), split = self.split)
        self.transform = transform

    def __len__(self):
        return len(self.ds.query_answer_pairs)
    
     def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        tokenizer = SpacySenticizer()
        query, document = self.query_answer_pairs(split=split)[idx]
        if document.id == MISSING_ID:
            logging.warning(f'Skipping {document.title} (missing ID)')
            return None
        key = (query, document.id)
        doc, example= None, None
        try:
            doc = self.loader.load_document(document.id)
            example = (key, tokenizer(doc.all_text))
        except ValueError as e:
            logging.warning(f'Skipping {document.id} ({e})')
            return None
        sents = tokenizer(doc.all_text)
        rel = (key, [False] * len(sents))
        for idx, s in enumerate(sents):
            if document.exact_answer in s:
                rel[1][idx] = True
        #evaluate
        mean_stats = defaultdict(list)
        
        int_rels = np.array(list(map(int, rel[1])))
        p = int_rels.sum()
        mean_stats['Average spans'] = p
        mean_stats['Random P@1'] = np.mean(int_rels)
        n = len(int_rels) - p
        mean_stats['Random R@3']=(1 - (n * (n - 1) * (n - 2)) / (N * (N - 1) * (N - 2)))
        numer = np.array([sp.comb(n, i) / (N - i) for i in range(0, n + 1)]) * p
        denom = np.array([sp.comb(N, i) for i in range(0, n + 1)])
        rr = 1 / np.arange(1, n + 2)
        rmrr = np.sum(numer * rr / denom)
        mean_stats['Random MRR']= (rmrr)
        if not any(rels):
            logging.warning(f'{doc_id} has no relevant answers')N = len(int_rels)
        for k, v in mean_stats.items():
            logging.info(f'{k}: {np.mean(v)}')
        return RelevanceExample(Query(query),  list(map(lambda s: Text(s,
                dict(docid=docid)), sents)), rel[1]) 
             


class KaggleReranker(pl.LightningModule):
    def __init__(self, options: KaggleEvaluationOptions):
        super().__init__()
        self.options = options
        self.evaluate_option = construct_t5(options)
        self.tokenizer = self.evaluate_option['tokenizer']
        self.reranker_evaluator = RerankerEvaluator(self.evaluate_option['reranker'], options.metrics)
        self.test_dataset = MyIterableDataset(self.options)
        self.batch_size = 4 # set it to 4 for now

    def test_dataloader():
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def test_step(self, batch, batch_idx):
        batch_example = batch 
        metric_result = dict()
        for metric in self.reranker_evaluator.evaluate(examples):
            metric_result[f'{metric.name:<{width}}for batch{batch_idx}'] = f'{metric.value:.5}'
        return metrix_result



       
        
   
       

        
       