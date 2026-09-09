import os
import csv
import h5py
import json
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from utils.trend import smooth_salience_scores

# MoSu and Mr. HiSum dataset class to load data and provide it to the model during training and testing
class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        self.split = split
        self.input_modalities = frozenset(getattr(
            cfg, 'input_modalities', ('visual', 'text', 'audio')
        ))
        self.salience_smoothing_window = getattr(
            cfg, 'salience_smoothing_window', 1
        )
        if (self.salience_smoothing_window < 1
                or self.salience_smoothing_window % 2 == 0):
            raise ValueError(
                'salience_smoothing_window must be a positive odd integer'
            )
        if cfg.dataset == 'mosu':
            split_path = os.path.join(cfg.data_dir, cfg.dataset, 'mosu_split.json')
            visual_path = os.path.join(cfg.data_dir, cfg.dataset, 'mosu_feat_visual_clip.h5')
            text_path = os.path.join(cfg.data_dir, cfg.dataset, 'mosu_feat_text_roberta.h5')
            audio_path = os.path.join(cfg.data_dir, cfg.dataset, 'mosu_feat_audio_ast.h5')
        
        elif cfg.dataset == 'mrhisum':
            split_path = os.path.join(cfg.data_dir, cfg.dataset, 'mrhisum_split.json')
            visual_path = os.path.join(cfg.data_dir, cfg.dataset, 'mrhisum_feat_visual_inceptionv3.h5')
            text_path = os.path.join(cfg.data_dir, cfg.dataset, 'mrhisum_feat_text_roberta.h5')
            audio_path = os.path.join(cfg.data_dir, cfg.dataset, 'mrhisum_feat_audio_ast.h5')

        elif cfg.dataset in ('summe', 'tvsum'):
            fold = getattr(cfg, 'fold', 0)
            protocol = getattr(cfg, 'split_protocol', 'tvt')
            split_name = (
                f'{cfg.dataset}_split_fold{fold}.json'
                if protocol == 'tvt'
                else f'{cfg.dataset}_split_tv_fold{fold}.json'
            )
            split_path = os.path.join(cfg.data_dir, cfg.dataset, split_name)
            visual_path = os.path.join(cfg.data_dir, cfg.dataset, f'{cfg.dataset}_feat_visual_googlenet.h5')
            text_path = os.path.join(cfg.data_dir, cfg.dataset, f'{cfg.dataset}_feat_text_roberta.h5')
            audio_path = os.path.join(cfg.data_dir, cfg.dataset, f'{cfg.dataset}_feat_audio_ast.h5')
        
        with open(split_path, 'r') as split_file:
            self.video_ids = json.load(split_file)[f'{split}_keys']

        # Optional: exclude a fixed set of video_ids from the TRAIN split only
        # (e.g. bottom-N% by some quality metric, precomputed offline). No-op
        # for every existing caller/script, since the default is None.
        exclude_ids_file = getattr(cfg, 'exclude_ids_file', None)
        if exclude_ids_file is not None and split == 'train':
            with open(exclude_ids_file, 'r') as f:
                exclude_ids = {line.strip() for line in f if line.strip()}
            self.video_ids = [vid for vid in self.video_ids if vid not in exclude_ids]

        self.salience_classification = cfg.loss_type == 'salience_cls'
        # Store paths and open lazily in __getitem__ (safe with num_workers > 0)
        self._gt_path = os.path.join(cfg.data_dir, cfg.dataset, f'{cfg.dataset}_gt.h5')
        self._trend_path = None
        if cfg.loss_type == 'trend':
            self._trend_path = os.path.join(
                cfg.data_dir, cfg.dataset, cfg.trend_label_file
            )
        self._trend_class_path = None
        if cfg.loss_type in ('trend_cls', 'trend_cls_mse'):
            self._trend_class_path = os.path.join(
                cfg.data_dir, cfg.dataset, cfg.trend_class_label_file
            )
        self._visual_path = visual_path
        self._text_path = text_path
        self._audio_path = audio_path
        self.gt_data = None
        self.trend_data = None
        self.trend_class_data = None
        self.visual_data = None
        self.text_data = None
        self.audio_data = None

        # Optional genre/mode (cluster) labels, either semantic genre from the
        # metadata CSV or data-driven gt_score-behavior clusters from
        # ana_mode.ipynb's output.
        self.cluster_map = {}
        if getattr(cfg, 'use_genre', False) and cfg.dataset == 'mosu':
            label_source = getattr(cfg, 'genre_label_source', 'metadata')
            if label_source == 'mode':
                cluster_path = os.path.join(cfg.data_dir, 'mosu', 'mosu_mode_clusters.csv')
                with open(cluster_path, 'r', encoding='utf-8-sig') as f:
                    reader = csv.DictReader(f)
                    self.cluster_map = {row['video_id']: int(row['mode_cluster']) for row in reader}
            else:
                metadata_path = os.path.join(cfg.data_dir, 'mosu', 'mosu_metadata.csv')
                with open(metadata_path, 'r', encoding='utf-8-sig') as f:
                    reader = csv.DictReader(f)
                    self.cluster_map = {row['video_id']: int(row['cluster_id']) for row in reader}

        # Optional: per-video training loss weight (e.g. down-weight
        # suspected-noisy videos instead of excluding them outright).
        # `video_id,weight` per line. TRAIN split only; None means "use 1.0
        # for everything", i.e. no-op for every existing caller/script.
        self.sample_weights = None
        sample_weights_file = getattr(cfg, 'sample_weights_file', None)
        if sample_weights_file is not None and split == 'train':
            self.sample_weights = {}
            with open(sample_weights_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    vid, weight = line.split(',')
                    self.sample_weights[vid] = float(weight)

    def __len__(self):
        return len(self.video_ids)

    def _open_files(self):
        """Open HDF5 files lazily (called after fork in each DataLoader worker)."""
        if self.visual_data is None:
            self.gt_data     = h5py.File(self._gt_path,     'r')
            if self._trend_path is not None:
                self.trend_data = h5py.File(self._trend_path, 'r')
            if self._trend_class_path is not None:
                self.trend_class_data = h5py.File(self._trend_class_path, 'r')
            self.visual_data = h5py.File(self._visual_path, 'r')
            self.text_data   = h5py.File(self._text_path,   'r')
            self.audio_data  = h5py.File(self._audio_path,  'r')

    def __getitem__(self, idx):
        self._open_files()
        data = {}
        video_id = self.video_ids[idx]
        data['video_id'] = video_id
        try:
            data['visual_feat'] = torch.Tensor(self.visual_data[video_id][...])
            data['text_feat'] = torch.Tensor(self.text_data[video_id][...])
            data['audio_feat'] = torch.Tensor(self.audio_data[video_id][...])
            for modality in ('visual', 'text', 'audio'):
                if modality not in self.input_modalities:
                    data[f'{modality}_feat'].zero_()
            
            original_score = torch.Tensor(self.gt_data[video_id]['gt_score'][...])
            if self.trend_data is not None:
                data['gt_score'] = torch.Tensor(self.trend_data[video_id][...])
                data['eval_gt_score'] = original_score
                data['trend_mask'] = torch.ones(data['gt_score'].shape[0], dtype=torch.bool)
                data['trend_mask'][-1] = False
            elif self.split == 'train' and self.salience_smoothing_window > 1:
                data['gt_score'] = torch.from_numpy(smooth_salience_scores(
                    original_score.numpy(), self.salience_smoothing_window
                ))
                data['eval_gt_score'] = original_score
            else:
                data['gt_score'] = original_score
            if self.salience_classification:
                data['salience_class'] = (original_score * 4).long().clamp(0, 3)
            if self.trend_class_data is not None:
                data['trend_class'] = torch.from_numpy(
                    self.trend_class_data[video_id][...].astype(np.int64)
                )
                data['trend_mask'] = torch.ones(
                    data['trend_class'].shape[0], dtype=torch.bool
                )
                data['trend_mask'][-1] = False
            data['mask'] = torch.ones(data['gt_score'].shape[0], dtype=torch.bool)
            if self.sample_weights is not None:
                data['sample_weight'] = torch.tensor(
                    self.sample_weights.get(video_id, 1.0), dtype=torch.float32
                )

            data['gt_summary'] = self.gt_data[video_id]['gt_summary'][...]
            data['change_points'] = self.gt_data[video_id]['change_points'][...]
            data['n_frames'] = int(data['visual_feat'].shape[0])
            data['picks'] = np.array([i for i in range(data['n_frames'])])
            if self.cluster_map:
                data['cluster_id'] = torch.tensor(self.cluster_map.get(video_id, 0), dtype=torch.long)
        except KeyError:
            return None
        return data

# Collate function to pad variable-length sequences in a batch
class CollateFn:
    def __call__(self, batch):
        batch = [item for item in batch if item is not None]
        data = {}
        data['video_id'] = [item['video_id'] for item in batch]
        
        data['visual_feat'] = pad_sequence([item['visual_feat'] for item in batch], batch_first=True, padding_value=0.0)
        data['text_feat'] = pad_sequence([item['text_feat'] for item in batch], batch_first=True, padding_value=0.0)
        data['audio_feat'] = pad_sequence([item['audio_feat'] for item in batch], batch_first=True, padding_value=0.0)
        
        data['gt_score'] = pad_sequence([item['gt_score'] for item in batch], batch_first=True, padding_value=0.0)
        data['mask'] = pad_sequence([item['mask'] for item in batch], batch_first=True, padding_value=0.0)
        if 'trend_mask' in batch[0]:
            data['trend_mask'] = pad_sequence([item['trend_mask'] for item in batch], batch_first=True, padding_value=0.0)
        if 'trend_class' in batch[0]:
            data['trend_class'] = pad_sequence(
                [item['trend_class'] for item in batch],
                batch_first=True,
                padding_value=1,
            )
        if 'eval_gt_score' in batch[0]:
            data['eval_gt_score'] = pad_sequence([item['eval_gt_score'] for item in batch], batch_first=True, padding_value=0.0)
        if 'salience_class' in batch[0]:
            data['salience_class'] = pad_sequence([item['salience_class'] for item in batch], batch_first=True, padding_value=0)
        
        data['gt_summary'] = [item['gt_summary'] for item in batch]
        data['change_points'] = [item['change_points'] for item in batch]
        data['n_frames'] = [item['n_frames'] for item in batch]
        data['picks'] = [item['picks'] for item in batch]
        if 'cluster_id' in batch[0]:
            data['cluster_id'] = torch.stack([item['cluster_id'] for item in batch])
        if 'sample_weight' in batch[0]:
            data['sample_weight'] = torch.stack([item['sample_weight'] for item in batch])
        return data