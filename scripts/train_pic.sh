date=250930
batchname=pic

export CUDA_VISIBLE_DEVICES=0
dataname=image_scale
pixelsize=2
color_type=white
iri_range=0.06
sample_range=1
render_scale=4
sample_exp=2
material="Aluminium"
C1_method="Kirchhoff"
res=32
image_path="./data/"

scaler=0.006
norm_factor=0.8
sigma=3.9
img_name=note
BRDF_image_path=${image_path}${img_name}".png"
blur_sigma=0.5
noise=0.0

python train_fabricating.py --resolution ${res} \
                --model_path ./output/${date}/${batchname}/sigma=${sigma}_blur_sigma=${blur_sigma}_${scaler}_${norm_factor}_${img_name}_pixelsize=${pixelsize}_${color_type}_${res}/  \
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
                --noise ${noise} \
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

export CUDA_VISIBLE_DEVICES=1
scaler=0.04
img_name=heart
BRDF_image_path=${image_path}${img_name}".png"

python train_fabricating.py --resolution ${res} \
                --model_path ./output/${date}/${batchname}/sigma=${sigma}_blur_sigma=${blur_sigma}_${scaler}_${norm_factor}_${img_name}_pixelsize=${pixelsize}_${color_type}_${res}/  \
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
                --noise ${noise} \
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

export CUDA_VISIBLE_DEVICES=2
scaler=0.01
img_name=lightning
BRDF_image_path=${image_path}${img_name}".png"

python train_fabricating.py --resolution ${res} \
                --model_path ./output/${date}/${batchname}/sigma=${sigma}_blur_sigma=${blur_sigma}_${scaler}_${norm_factor}_${img_name}_pixelsize=${pixelsize}_${color_type}_${res}/  \
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
                --noise ${noise} \
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
scaler=0.04
img_name=rabbit
BRDF_image_path=${image_path}${img_name}".png"

python train_fabricating.py --resolution ${res} \
                --model_path ./output/${date}/${batchname}/sigma=${sigma}_blur_sigma=${blur_sigma}_${scaler}_${norm_factor}_${img_name}_pixelsize=${pixelsize}_${color_type}_${res}/  \
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
                --noise ${noise} \
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
scaler=0.04
img_name=butterfly
BRDF_image_path=${image_path}${img_name}".png"

python train_fabricating.py --resolution ${res} \
                --model_path ./output/${date}/${batchname}/sigma=${sigma}_blur_sigma=${blur_sigma}_${scaler}_${norm_factor}_${img_name}_pixelsize=${pixelsize}_${color_type}_${res}/  \
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
                --noise ${noise} \
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