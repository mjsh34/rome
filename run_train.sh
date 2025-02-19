export LOCAL_RANK=1
python3 -m torch.distributed.launch --master_port 12345 --nproc_per_node=1 train.py \
--experiment_name test_train \
--dataset_name voxceleb2hq_pairs \
--model_name parametric_avatar \
--num_gpus 1 \
--max_epochs 1500 \
--data_root datasets/VoxCeleb2HQ \
--keys_name 'keys_all' \
--project_dir parametric_results \
--image_size 256 \
--batch_size 4 \
--test_batch_size 4 \
--augment_geometric_source False \
--align_source True \
--align_scale 1.25 \
--autoenc_cat_alphas True \
--autoenc_num_groups 4 \
--neural_texture_channels 8 \
--unet_num_channels 64 \
--unet_max_channels 512 \
--unet_num_groups 4 \
--unet_use_normals_cond \
--unet_pred_mask \
--norm_layer_type gn \
--conv_layer_type ws_conv \
--deform_norm_layer_type gn \
--conv_layer_type ws_conv \
--spn_apply_to_dis \
--renderer_type hard_mesh \
--use_random_uniform_background True \
--train_deferred_neural_rendering True \
--use_neck_dir False \
--use_gaze_dir False \
--use_mesh_deformations True \
--use_scalp_deforms \
--use_neck_deforms \
--detach_silhouettes True \
--output_unet_deformer_feats 32 \
--harmonize_deform_input True \
--adversarial_weight 0.1 \
--feature_matching_weight 1.0 \
--vgg19_weight 1.0 \
--vggface_weight 0.1 \
--unet_seg_weight 10.0 \
--unet_seg_type dice \
--seg_hair_weight 10.0 \
--seg_neck_weight 1.0 \
--seg_type mse \
--laplacian_reg_weight 10.0 \
--keypoints_matching_weight 1.0 \
--eye_closure_weight 0.0 \
--lip_closure_weight 0.0 \
--gen_lr 1e-4 \
--dis_lr 4e-4 \
--use_amp False \
--amp_opt_level O0 \
--logging_freq 1000 \
--test_freq 1 \
--test_visual_freq 2 \
--visuals_freq 5000 \
--redirect_print_to_file False \
--chamfer_weight 0.01 \
--chamfer_same_num_points \
--use_separate_seg_unet False \
--subdivide_mesh False \
--deform_along_normals \
--deca_path DECA \
--rome_data_dir data \
--face_parsing_path face-parsing.PyTorch
