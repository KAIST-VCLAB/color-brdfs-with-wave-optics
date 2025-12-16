date=250929
batchname=lightning

export CUDA_VISIBLE_DEVICES=2
dataname=image_scale
pixelsize=0.8
color_type=red
iri_range=0.06
sample_range=0.5
render_scale=4
sample_exp=2
material="Aluminium"
C1_method="Kirchhoff"
res=32
image_path="./data/"

scaler=0.1
norm_factor=0.8
sigma=2.6
img_name=lightning
BRDF_image_path=${image_path}${img_name}".png"
blur_sigma=0.1

python train_fabricating.py --resolution ${res} \
                --model_path ./output/${date}/${batchname}/range=${sample_range}_sigma=${sigma}_blur_sigma=${blur_sigma}_${scaler}_${norm_factor}_${img_name}_pixelsize=${pixelsize}_${color_type}_${res}/  \
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
                --BRDF_image_path ${BRDF_image_path} \
                --color_type ${color_type} \
                --sample_range ${sample_range} \
                --sample_exp ${sample_exp} \
                --material ${material} \
                --fix_scaler \
                --intensity_scaler ${scaler} \
                --norm_hmap \
                --pred_gamma 2.2 \
                --norm_factor_hmap ${norm_factor} &

export CUDA_VISIBLE_DEVICES=3
color_type=blue

python train_fabricating.py --resolution ${res} \
                --model_path ./output/${date}/${batchname}/range=${sample_range}_sigma=${sigma}_blur_sigma=${blur_sigma}_${scaler}_${norm_factor}_${img_name}_pixelsize=${pixelsize}_${color_type}_${res}/  \
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
                --BRDF_image_path ${BRDF_image_path} \
                --color_type ${color_type} \
                --sample_range ${sample_range} \
                --sample_exp ${sample_exp} \
                --material ${material} \
                --fix_scaler \
                --intensity_scaler ${scaler} \
                --norm_hmap \
                --pred_gamma 2.2 \
                --norm_factor_hmap ${norm_factor} &
                
export CUDA_VISIBLE_DEVICES=4
color_type=green

python train_fabricating.py --resolution ${res} \
                --model_path ./output/${date}/${batchname}/range=${sample_range}_sigma=${sigma}_blur_sigma=${blur_sigma}_${scaler}_${norm_factor}_${img_name}_pixelsize=${pixelsize}_${color_type}_${res}/  \
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
                --BRDF_image_path ${BRDF_image_path} \
                --color_type ${color_type} \
                --sample_range ${sample_range} \
                --sample_exp ${sample_exp} \
                --material ${material} \
                --fix_scaler \
                --intensity_scaler ${scaler} \
                --norm_hmap \
                --pred_gamma 2.2 \
                --norm_factor_hmap ${norm_factor} &