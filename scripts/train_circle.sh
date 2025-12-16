date=251001
batchname=circle

export CUDA_VISIBLE_DEVICES=0
dataname=diffuse
pixelsize=0.8
color_type=disk_iri
iri_range=0.09
sample_range=1
render_scale=4
sample_exp=2
material="Aluminium"
C1_method="Kirchhoff"
res=32

scaler=0.02
norm_factor=0.8
sigma=2.6
blur_sigma=0.1

python train_fabricating.py --resolution ${res} \
                --model_path ./output/${date}/${batchname}/iri_range=${iri_range}_sigma=${blur_sigma}_${scaler}_${norm_factor}_${dataname}_pixelsize=${pixelsize}_${color_type}_${res}/  \
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
                --blur_sigma ${blur_sigma} \
                --sigma ${sigma} \
                --pixelsize ${pixelsize} \
                --BRDF_name  ${dataname} \
                --color_type ${color_type} \
                --sample_range ${sample_range} \
                --fix_scaler \
                --norm_hmap \
                --norm_factor_hmap ${norm_factor}  \
                --intensity_scaler ${scaler} \
                --sample_exp ${sample_exp} \
                --material ${material} \
                --pred_gamma 2.2  &

export CUDA_VISIBLE_DEVICES=1
color_type=inv_disk_iri

python train_fabricating.py --resolution ${res} \
                --model_path ./output/${date}/${batchname}/iri_range=${iri_range}_sigma=${blur_sigma}_${scaler}_${norm_factor}_${dataname}_pixelsize=${pixelsize}_${color_type}_${res}/  \
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
                --blur_sigma ${blur_sigma} \
                --sigma ${sigma} \
                --pixelsize ${pixelsize} \
                --BRDF_name  ${dataname} \
                --color_type ${color_type} \
                --sample_range ${sample_range} \
                --fix_scaler \
                --norm_hmap \
                --norm_factor_hmap ${norm_factor}  \
                --intensity_scaler ${scaler} \
                --sample_exp ${sample_exp} \
                --material ${material} \
                --pred_gamma 2.2  