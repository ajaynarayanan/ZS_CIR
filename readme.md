## Language-only Efficient Training of Zero-shot Composed Image Retrieval
---
* Fall 2023, CSE 597: Vision and Language Course Project offered at Penn State University
* Task: Zero-shot Composed Image Retrieval

### Dataset and Environment Setup

* Please refer to [LinCIR's README](lincir/README.md) for dataset preparation, environment setup, and a description of files.

### Training
* Script to train LinCIR (uses CC3M, Stable Diffusion Prompts datasets) model.

```
python -m torch.distributed.run --nproc_per_node 1 --nnodes 1 --node_rank 0 \
--master_addr localhost --master_port 5100 train_phi.py \
--batch_size 512 \
--output_dir <path_to_outputs>/outputs \
--cirr_dataset_path <path_to_cirr_dataset>/lincir/datasets/CIRR \
--mixed_precision fp16 \
--clip_model_name large \
--validation_steps 500 \
--checkpointing_steps 500 \
--seed 12345 \
--lr_scheduler constant_with_warmup --lr_warmup_steps 0 \
--max_train_steps 30000 
```

### Evaluation 

* Script to run testing process in FashionIQ validation dataset with re-ranking enabled.
```
CUDA_VISIBLE_DEVICES=1 python validate.py \
--eval-type phi \
--dataset fashioniq \
--dataset-path <path_to_datasets>/datasets/fashion-iq \
--phi-checkpoint-name <path_to_outputs>/outputs/checkpoints/phi_000030000.pt \
--clip_model_name large \
--enable_re_rank \
--k1 20 \
--k2 5 \
--lambda_value 0.3
```

* Script to run testing process in CIRR test set  with re-ranking enabled.
```
CUDA_VISIBLE_DEVICES=1 python generate_test_submission.py \
--eval-type phi \
--dataset cirr \
--dataset-path <path_to_datasets>/datasets/CIRR \
--phi-checkpoint-name <path_to_outputs>/outputs/checkpoints/phi_000030000.pt \
--clip_model_name large \
--submission-name cirlo_results_rerank \
--enable_re_rank \
--k1 20 \
--k2 5 \
--lambda_value 0.3
```



## References:
```
@article{gu2023language,
  title={Language-only Efficient Training of Zero-shot Composed Image Retrieval},
  author={Gu, Geonmo and Chun, Sanghyuk and Kim, Wonjae and Kang, Yoohoon and Yun, Sangdoo},
  journal={arXiv preprint arXiv:2312.01998},
  year={2023}
}

@inproceedings{zhong2017re,
  title={Re-ranking person re-identification with k-reciprocal encoding},
  author={Zhong, Zhun and Zheng, Liang and Cao, Donglin and Li, Shaozi},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1318--1327},
  year={2017}
}
```

The repository is adapted from [LinCIR](https://github.com/navervision/lincir/tree/master) and [SEARLE](https://github.com/miccunifi/SEARLE). 