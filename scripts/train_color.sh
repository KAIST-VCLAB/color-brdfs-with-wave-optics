date=250928
batchname=uniform

export CUDA_VISIBLE_DEVICES=0
dataname=uniform
pixelsize=0.8
color_type=red
iri_range=0.06
sample_range=0.15
render_scale=4
sample_exp=1
material="Aluminium"
C1_method="Kirchhoff"
res=32

scaler=0.16
norm_factor=1.2
sigma=2.6
blur_sigma=0.13
noise=0.0

python train_fabricating.py --resolution ${res} \
                --model_path ./output/${date}/${batchname}/sigma=${sigma}_blur_sigma=${blur_sigma}_${scaler}_${norm_factor}_${dataname}_pixelsize=${pixelsize}_${color_type}_${res}/  \
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
                --noise ${noise} \
                --pixelsize ${pixelsize} \
                --BRDF_name  ${dataname} \
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
color_type=blue

python train_fabricating.py --resolution ${res} \
                --model_path ./output/${date}/${batchname}/sigma=${sigma}_blur_sigma=${blur_sigma}_${scaler}_${norm_factor}_${dataname}_pixelsize=${pixelsize}_${color_type}_${res}/  \
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
                --noise ${noise} \
                --pixelsize ${pixelsize} \
                --BRDF_name  ${dataname} \
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
color_type=green

python train_fabricating.py --resolution ${res} \
                --model_path ./output/${date}/${batchname}/sigma=${sigma}_blur_sigma=${blur_sigma}_${scaler}_${norm_factor}_${dataname}_pixelsize=${pixelsize}_${color_type}_${res}/  \
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
                --noise ${noise} \
                --pixelsize ${pixelsize} \
                --BRDF_name  ${dataname} \
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
color_type=yellow

python train_fabricating.py --resolution ${res} \
                --model_path ./output/${date}/${batchname}/sigma=${sigma}_blur_sigma=${blur_sigma}_${scaler}_${norm_factor}_${dataname}_pixelsize=${pixelsize}_${color_type}_${res}/  \
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
                --noise ${noise} \
                --pixelsize ${pixelsize} \
                --BRDF_name  ${dataname} \
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
color_type=orange

python train_fabricating.py --resolution ${res} \
                --model_path ./output/${date}/${batchname}/sigma=${sigma}_blur_sigma=${blur_sigma}_${scaler}_${norm_factor}_${dataname}_pixelsize=${pixelsize}_${color_type}_${res}/  \
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
                --noise ${noise} \
                --pixelsize ${pixelsize} \
                --BRDF_name  ${dataname} \
                --color_type ${color_type} \
                --sample_range ${sample_range} \
                --sample_exp ${sample_exp} \
                --material ${material} \
                --fix_scaler \
                --intensity_scaler ${scaler} \
                --norm_hmap \
                --pred_gamma 2.2 \
                --norm_factor_hmap ${norm_factor} &

export CUDA_VISIBLE_DEVICES=5
color_type=cyan

python train_fabricating.py --resolution ${res} \
                --model_path ./output/${date}/${batchname}/sigma=${sigma}_blur_sigma=${blur_sigma}_${scaler}_${norm_factor}_${dataname}_pixelsize=${pixelsize}_${color_type}_${res}/  \
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
                --noise ${noise} \
                --pixelsize ${pixelsize} \
                --BRDF_name  ${dataname} \
                --color_type ${color_type} \
                --sample_range ${sample_range} \
                --sample_exp ${sample_exp} \
                --material ${material} \
                --fix_scaler \
                --intensity_scaler ${scaler} \
                --norm_hmap \
                --pred_gamma 2.2 \
                --norm_factor_hmap ${norm_factor} &

export CUDA_VISIBLE_DEVICES=6
color_type=purple

python train_fabricating.py --resolution ${res} \
                --model_path ./output/${date}/${batchname}/sigma=${sigma}_blur_sigma=${blur_sigma}_${scaler}_${norm_factor}_${dataname}_pixelsize=${pixelsize}_${color_type}_${res}/  \
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
                --noise ${noise} \
                --pixelsize ${pixelsize} \
                --BRDF_name  ${dataname} \
                --color_type ${color_type} \
                --sample_range ${sample_range} \
                --sample_exp ${sample_exp} \
                --material ${material} \
                --fix_scaler \
                --intensity_scaler ${scaler} \
                --norm_hmap \
                --pred_gamma 2.2 \
                --norm_factor_hmap ${norm_factor} &