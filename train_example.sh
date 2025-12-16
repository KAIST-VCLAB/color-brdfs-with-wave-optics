date=250926
batchname=train_example

export CUDA_VISIBLE_DEVICES=0
dataname=anti_mirror
pixelsize=0.8
color_type=white
iri_range=0.06
sample_range=1
render_scale=4
sample_exp=2
material="Aluminium"
C1_method="Kirchhoff"
res=32

scaler=0.05
norm_factor=0.3
sigma=2.6
blur_sigma=0.13

python train_fabricating.py --resolution ${res} \
                --model_path ./output/${date}/${batchname}/sample=${sample_range}_blur=${blur_sigma}/${scaler}_${norm_factor}_${dataname}_pixelsize=${pixelsize}_${color_type}/  \
                --hmap_lr_init 0.00005 \
                --hmap_lr_final 0.000005 \
                --hmap_max_steps 500000 \
                --save_interval 10000 \
                --test_interval 10000 \
                --iteration_num 500000 \
                --iri_range ${iri_range} \
                --render_scale ${render_scale} \
                --C1 ${C1_method} \
                --C3 nonparaxial \
                --sigma ${sigma} \
                --fix_scaler \
                --intensity_scaler ${scaler} \
                --norm_factor_hmap ${norm_factor} \
                --blur_sigma ${blur_sigma} \
                --pixelsize ${pixelsize} \
                --BRDF_name  ${dataname} \
                --color_type ${color_type} \
                --sample_range ${sample_range} \
                --sample_exp ${sample_exp} \
                --material ${material} \
                --pred_gamma 2.2 