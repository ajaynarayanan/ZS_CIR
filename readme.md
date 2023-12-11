
### Script to run training process 
python -m torch.distributed.run --nproc_per_node 1 --nnodes 1 --node_rank 0 \
--master_addr localhost --master_port 5100 train_phi.py \
--batch_size 512 \
--output_dir /home/mdl/afs6372/courses/fall23/cse597_vision_lang/CSE597_Project/v100 \
--cirr_dataset_path /path/to/cir_datasets/CIRR \
--mixed_precision fp16 \
--clip_model_name large \
--validation_steps 500 \
--checkpointing_steps 500 \
--seed 12345 \
--lr_scheduler constant_with_warmup --lr_warmup_steps 0 \
--max_train_steps 30000 

### Script to run testing process
CUDA_VISIBLE_DEVICES=1 python validate.py \
--eval-type phi \
--dataset fashioniq \
--dataset-path /home/mdl/afs6372/courses/fall23/cse597_vision_lang/CSE597_Project/lincir/datasets/fashion-iq \
--phi-checkpoint-name /home/mdl/afs6372/courses/fall23/cse597_vision_lang/CSE597_Project/outputs/checkpoints/phi_latest.pt \
--clip_model_name large