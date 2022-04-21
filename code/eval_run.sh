for seed in 666 42 1024;
do
python eval_model.py --env hopper --dataset medium --model_type dt --seed $seed --pretrained_lm gpt2  --outdir "checkpoints/gpt2_medium_hopper_$seed" --dropout 0.2 --share_input_output_proj &
done
for seed in 666 42 1024;
do
python eval_model.py --env walker2d --dataset medium --model_type dt --seed $seed  --pretrained_lm gpt2  --outdir "checkpoints/gpt2_medium_walker_$seed" --dropout 0.2 --share_input_output_proj &
done
for seed in 666 42 1024;
do
python eval_model.py --env halfcheetah --dataset medium --model_type dt --seed $seed  --pretrained_lm gpt2  --outdir "checkpoints/chibiv2_kmeans__medium_positions_halfcheetah_$seed" --dropout 0.2 --share_input_output_proj &
done
wait
