python3 explainer_swin.py --backbone_name vit --target_model_name vit --num_players 16 --use_conv '' --freeze_backbone none --label better_load
python3 explainer_swin.py --backbone_name vit --target_model_name vit --num_players 16 --use_conv '' --freeze_backbone all --label better_load
python3 explainer_swin.py --backbone_name vit --target_model_name vit --num_players 16 --use_conv '' --freeze_backbone except_last_two --label better_load


python3 explainer_swin.py --backbone_name t2t_vit --target_model_name t2t_vit --num_players 16 --use_conv '' --freeze_backbone none --label better_load
python3 explainer_swin.py --backbone_name t2t_vit --target_model_name t2t_vit --num_players 16 --use_conv '' --freeze_backbone all --label better_load
python3 explainer_swin.py --backbone_name t2t_vit --target_model_name t2t_vit --num_players 16 --use_conv '' --freeze_backbone except_last_two --label better_load

python3 explainer_swin.py --backbone_name swin --target_model_name swin --num_players 16 --use_conv '' --freeze_backbone none --label better_load
python3 explainer_swin.py --backbone_name swin --target_model_name swin --num_players 16 --use_conv '' --freeze_backbone all --label better_load
python3 explainer_swin.py --backbone_name swin --target_model_name swin --num_players 16 --use_conv '' --freeze_backbone except_last_two --label better_load
