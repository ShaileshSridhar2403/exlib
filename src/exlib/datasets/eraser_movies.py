import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import os
import requests
import tarfile
import jsonlines


class EraserMovies(Dataset): 
    id2label = {
        0: 'NEG',
        1: 'POS'
    }
    label2id = {
        'NEG': 0,
        'POS': 1
    }
    def __init__(self, data_dir, split='train', 
                 transform=None,
                 download=False,
                 data_size=-1):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform

        if download: 
            # Create directory structure
            os.makedirs(data_dir, exist_ok=True)  # exist_ok=True will not raise an exception if the directory exists

            # Download file
            url = 'https://www.eraserbenchmark.com/zipped/movies.tar.gz'
            response = requests.get(url)

            # tar_gz_path = os.path.join(data_dir, 'movies.tar.gz')
            tar_gz_path = data_dir
            with open(tar_gz_path, 'wb') as f:
                f.write(response.content)

            # Extract tar.gz file
            movies_folder = os.path.join(data_dir, 'movies')
            os.makedirs(movies_folder, exist_ok=True)

            with tarfile.open(tar_gz_path, 'r:gz') as file:
                file.extractall(path=movies_folder)

        # self.movies_folder = os.path.join(data_dir, 'movies')
        self.movies_folder = data_dir
        self.docs_folder = os.path.join(self.movies_folder, 'docs')

        # X = []
        passages = []
        queries = []
        y = []
        evidences = []
        with jsonlines.open(os.path.join(self.movies_folder, 
                                         f'{split}.jsonl')) as reader:
            for obj in reader:
                filename = obj['annotation_id']
                filepath = os.path.join(self.docs_folder, filename)
                query = obj['query']
                label = self.label2id[obj['classification']]
                if data_size != -1 and y.count(label) >= data_size:
                    continue
                with open(filepath, 'rt') as input_file:
                    passage = input_file.read()
                # text = f'{passage} || {query}'
                # if transform:
                #     text = transform(text)
                # X.append(text)
                passages.append(passage)
                queries.append(query)
                y.append(label)
                evidences.append(obj['evidences'])

        # self.X = X
        self.passages = passages
        self.queries = queries
        self.y = y
        self.evidences = evidences

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        batch = {
            'passage': self.passages[idx],
            'query': self.queries[idx],
            'label': self.y[idx],
            'evidences': self.evidences[idx]
        }
        if self.transform:
            batch = self.transform(batch)
        return batch
        # return self.X[idx], self.y[idx]


if __name__ == '__main__':
    dataset = EraserMovies('/nlp/data/weiqiuy/datasets')
