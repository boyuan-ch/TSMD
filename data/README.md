# Dataset layout

Download the feature and annotation files from the TripleSumm releases:

- [MoSu](https://huggingface.co/datasets/hminjeong/TripleSumm-MoSu)
- [Mr. HiSum](https://huggingface.co/datasets/hminjeong/TripleSumm-Mr.HiSum)

Arrange them as follows:

```text
data/
├── mosu/
│   ├── mosu_gt.h5
│   ├── mosu_feat_visual_clip.h5
│   ├── mosu_feat_audio_ast.h5
│   ├── mosu_feat_text_roberta.h5
│   └── mosu_split.json
└── mrhisum/
    ├── mrhisum_gt.h5
    ├── mrhisum_feat_visual_googlenet.h5
    ├── mrhisum_feat_audio_ast.h5
    ├── mrhisum_feat_text_roberta.h5
    └── mrhisum_split.json
```

Each feature file stores one `(T, D)` dataset per video identifier. Ground-truth
files contain per-video `gt_score`, `gt_summary`, and `change_points` datasets.
The split JSON enumerates training, validation, and test video identifiers.

MoSu feature files can have a damaged HDF5 key index. The loader therefore uses
the split JSON for enumeration and accesses video keys directly. A small number
of unreadable records are skipped during collation.

Dataset files are excluded from Git by `.gitignore`.
