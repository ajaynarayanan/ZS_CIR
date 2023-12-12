
### Script to run training process (uses CC3M, Stable Diffusion Prompts datasets) 
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

### Script to run testing process in FashionIQ validation dataset
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


### Script to run testing process in CIRR test set
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

