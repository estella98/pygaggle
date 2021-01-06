from collections import OrderedDict, defaultdict
from typing import Optional, List
from copy import deepcopy
from pathlib import Path
import json
import logging
import scipy.special as sp
import numpy as np
from pygaggle.data import LitReviewDataset, Cord19DocumentLoader, RelevanceExample
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
from pygaggle.rerank.base import Reranker, Query, Text
from pygaggle.rerank.bm25 import Bm25Reranker
from pygaggle.model.tokenize import SpacySenticizer
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
                            metric_names,
                            QueryDocumentBatch
                            )
from pygaggle.model.evaluate import MetricAccumulator
from pygaggle.settings import Cord19Settings
from torch.utils.data import Dataset, DataLoader

MISSING_ID = '<missing>'
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
    
    
class RerankDataset(Dataset):
    def __init__(self, example:RelevanceExample, reranker_evaluator:RerankerEvaluator):
        self.reranker_evaluator = reranker_evaluator
        self.query = example.query
        self.texts = example.documents
        self.batch_inputs = QueryDocumentBatch(query=self.query, documents=deepcopy(self.texts))
        self.model_inputs = list(self.reranker_evaluator.reranker.tokenizer.traverse_query_document(self.batch_inputs))

    def __len__(self):
        return len(self.model_inputs)
    
    def __getitem__(self, idx):
        batch = self.model_inputs[idx]
        device = self.reranker_evaluator.reranker.device
        model = self.reranker_evaluator.reranker.model
        input_ids = batch.output['input_ids'].to(device)
        attn_mask = batch.output['attention_mask'].to(device)
        decode_ids = torch.full((input_ids.size(0), 1),
                            model.config.decoder_start_token_id,
                            dtype=torch.long).to(input_ids.device)
        past = model.get_encoder()(input_ids, attention_mask=attn_mask)
        batch_model_input = model.prepare_inputs_for_generation(
            decode_ids,
            past=past,
            attention_mask=attn_mask,
            use_cache=True)
        return batch_model_input



class MyIterableDataset(Dataset):
    def __init__(self, options:KaggleEvaluationOptions, reranker_evaluator:RerankerEvaluator, transform=None):
        """
        Args:
            options: Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ds = LitReviewDataset.from_file(str(options.dataset))
        self.examples = self.ds.to_senticized_dataset(str(options.index_dir),
                                        split=options.split)
        self.reranker_evaluator = reranker_evaluator
        self.batch_inputs = self.serialize()

    def serialize(self):
        batch_inputs = [QueryDocumentBatch(query=e.query, documents=deepcopy(e.documents)) for e in self.examples]
        return list(self.reranker_evaluator.reranker.tokenizer.traverse_query_documents_whole(batch_inputs))

    def __len__(self):
        return len(self.batch_inputs)
    
    def __getitem__(self, idx):
        batch = self.batch_inputs[idx]
        device = self.reranker_evaluator.reranker.device
        model = self.reranker_evaluator.reranker.model
        input_ids = batch.output['input_ids'].to(device)
        attn_mask = batch.output['attention_mask'].to(device)
        decode_ids = torch.full((input_ids.size(0), 1),
                            model.config.decoder_start_token_id,
                            dtype=torch.long).to(input_ids.device)
        past = model.get_encoder()(input_ids, attention_mask=attn_mask)
        batch_model_input = model.prepare_inputs_for_generation(
            decode_ids,
            past=past,
            attention_mask=attn_mask,
            use_cache=True)
        return batch_model_input

        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        # tokenizer = SpacySenticizer()
        # query, document = self.query_answer_pairs[idx]
        # if document.id == MISSING_ID:
        #     logging.warning(f'Skipping {document.title} (missing ID)')
        #     return None
        # key = (query, document.id)
        # doc, example= None, None
        # try:
        #     doc = self.loader.load_document(document.id)
        #     example = (key, tokenizer(doc.all_text))
        # except ValueError as e:
        #     logging.warning(f'Skipping {document.id} ({e})')
        #     return None
        # sents = tokenizer(doc.all_text)
        # rel = (key, [False] * len(sents))
        # for idx, s in enumerate(sents):
        #     if document.exact_answer in s:
        #         rel[1][idx] = True
        # #evaluate
        # mean_stats = defaultdict(list)
        # int_rels = np.array(list(map(int, rel[1])))
        # p = int_rels.sum()
        # mean_stats['Average spans'] = p
        # mean_stats['Random P@1'] = np.mean(int_rels)
        # n = len(int_rels) - p
        # N = len(int_rels)
        # mean_stats['Random R@3']=(1 - (n * (n - 1) * (n - 2)) / (N * (N - 1) * (N - 2)))
        # numer = np.array([sp.comb(n, i) / (N - i) for i in range(0, n + 1)]) * p
        # denom = np.array([sp.comb(N, i) for i in range(0, n + 1)])
        # rr = 1 / np.arange(1, n + 2)
        # rmrr = np.sum(numer * rr / denom)
        # mean_stats['Random MRR']= (rmrr)
        # if not any(rel[1]):
        #     logging.warning(f'{document.id} has no relevant answers')
        # return RelevanceExample(Query(query),  list(map(lambda s: Text(s,
        #         dict(docid=document.id)), sents)), rel[1]) 
             
# test_dataset = MyIterableDataset(options.dataset, options.split, options.index_dir)
class KaggleRerankerData(pl.LightningDataModule):
    def __init__(self, options: KaggleEvaluationOptions, example:RelevanceExample, pl_module):
        super().__init__()
        self.example_data = RerankDataset(example,pl_module)
    def train_dataloader(self):
        return DataLoader(self.example_data, batch_size=options.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

class KaggleReranker(pl.LightningModule):
  
    def __init__(self, options: KaggleEvaluationOptions):
        super().__init__()
        self.model = self.construct_t5(options)
        self.reranker_evaluator = RerankerEvaluator(self.model, metric_names())

    def forward(self, examples):
        self.reranker_evaluator.evaluate(examples)

    #load pre_trained model
    def construct_t5(self, options: KaggleEvaluationOptions) -> Reranker:
        cached_model_loader = CachedT5ModelLoader(SETTINGS.t5_model_dir,
                                    SETTINGS.cache_dir,
                                    'ranker',
                                    SETTINGS.t5_model_type,
                                    SETTINGS.flush_cache)
        device = torch.device(options.device)
        cached_model = cached_model_loader.load().to(device).eval()
        tokenizer = MonoT5.get_tokenizer(options.model_name,
                                        do_lower_case=options.do_lower_case,
                                        batch_size=options.batch_size)
        model= MonoT5(cached_model,tokenizer)
        return model

    # in the future, we should migrate the training, optimization step into here
    def training_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        pass

    def test_step(self, batch, batch_idx):
        batch_example = batch 
        metric_result = dict()
        for metric in self.reranker_evaluator.evaluate(batch_example):
            metric_result[f'{metric.name:<{width}}for batch{batch_idx}'] = f'{metric.value:.5}'
        return metrix_result
       






# def test(options):
#     model_reranker = KaggleReranker(options)
#     #examples = MyIterableDataset(options.dataset, options.split, options.index_dir)
#     ds = LitReviewDataset.from_file(str(options.dataset))
#     examples = ds.to_senticized_dataset(str(options.index_dir),
#                                         split=options.split)
#     #model_reranker.reranker.evaluate(examples)
#     try_out = model_reranker.eval().cuda(device = 0)
    # output = try_out(examples)
 # select between different gpu :
       #https://pytorch-lightning.readthedocs.io/en/stable/multi_gpu.html
       