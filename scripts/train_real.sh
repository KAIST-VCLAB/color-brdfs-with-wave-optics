date=250927
batchname=real

export CUDA_VISIBLE_DEVICES=0
dataname=aniso_anti_mirror
pixelsize=2
color_type=white
iri_range=0.06
sample_range=1
render_scale=4
sample_exp=2
material="Aluminium"
C1_method="Kirchhoff"
res=32

scaler=0.006
norm_factor=0.8
sigma=2.6
blur_sigma=0.65
noise=0.075

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
                --noise ${noise} \
                --sigma ${sigma} \
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
                --norm_factor_hmap ${norm_factor} 

export CUDA_VISIBLE_DEVICES=0
dataname=anti_mirror
pixelsize=2
color_type=white
iri_range=0.06
sample_range=1
render_scale=4
sample_exp=2
material="Aluminium"
C1_method="Kirchhoff"
res=32

scaler=0.01
norm_factor=0.8
sigma=2.6
blur_sigma=0.65
noise=0.075

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
                --noise ${noise} \
                --sigma ${sigma} \
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
                --norm_factor_hmap ${norm_factor} 

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

scaler=0.01
norm_factor=0.8
sigma=3.9
img_name=note
BRDF_image_path=${image_path}${img_name}".png"
blur_sigma=0.65
noise=0.075

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
                --norm_factor_hmap ${norm_factor} 

export CUDA_VISIBLE_DEVICES=0
dataname=image_scale
pixelsize=1.5
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
sigma=2.6
img_name=lightning
BRDF_image_path=${image_path}${img_name}".png"
blur_sigma=0.5
noise=0.075

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
                --norm_factor_hmap ${norm_factor} 

export CUDA_VISIBLE_DEVICES=0
dataname=image_inverse
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

scaler=0.02
norm_factor=0.8
sigma=5.2
img_name=batman
BRDF_image_path=${image_path}${img_name}".png"
blur_sigma=0.65
noise=0.075

python train_fabricating.py --resolution ${res} \
                --model_path ./output/${date}/${batchname}/sigma=${sigma}_blur_sigma=${blur_sigma}_${scaler}_${norm_factor}_inverse_${img_name}_pixelsize=${pixelsize}_${color_type}_${res}/  \
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
                --norm_factor_hmap ${norm_factor} 

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

scaler=0.01
norm_factor=0.8
sigma=5.2
img_name=batman
BRDF_image_path=${image_path}${img_name}".png"
blur_sigma=0.65
noise=0.075

python train_fabricating.py --resolution ${res} \
                --model_path ./output/${date}/${batchname}/sigma=${sigma}_blur_sigma=${blur_sigma}_${scaler}_${norm_factor}_inverse_${img_name}_pixelsize=${pixelsize}_${color_type}_${res}/  \
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
                --norm_factor_hmap ${norm_factor} 

export CUDA_VISIBLE_DEVICES=0
pixelsize=2
sample_range=1
render_scale=4
sample_exp=2
material="Aluminium"
C1_method="Kirchhoff"
res=32

norm_factor=0.8
blur_sigma=0.65
noise=0.075

sigma=2.6
dataname=diffuse
color_type=disk_iri
iri_range=0.09

scaler=0.03

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
                --noise ${noise} \
                --sigma ${sigma} \
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
                --norm_factor_hmap ${norm_factor} 

export CUDA_VISIBLE_DEVICES=0
sample_range=1
render_scale=4
sample_exp=2
material="Aluminium"
C1_method="Kirchhoff"
res=32

norm_factor=0.8
blur_sigma=0.65
noise=0.075

sigma=2.6

pixelsize=2
dataname=diffuse
color_type=inv_disk_iri
iri_range=0.09

scaler=0.03

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
                --noise ${noise} \
                --sigma ${sigma} \
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
                --norm_factor_hmap ${norm_factor} 

export CUDA_VISIBLE_DEVICES=0
dataname=image_scale
pixelsize=1.5
color_type=white
iri_range=0.06
sample_range=1
render_scale=4
sample_exp=2
material="Aluminium"
C1_method="Kirchhoff"
res=32
image_path="/data/volume1/zyx45889/yx.zeng/waveGS/result/brdf/"

scaler=0.006
norm_factor=0.8
sigma=3.9
img_name=dots_color_close
BRDF_image_path=${image_path}${img_name}".png"
blur_sigma=0.65
noise=0.075

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
                --BRDF_image_scale 2.5 \
                --norm_hmap \
                --pred_gamma 2.2 \
                --norm_factor_hmap ${norm_factor} &

export CUDA_VISIBLE_DEVICES=0
img_name=dots_color_close_inv
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
                --BRDF_image_scale 2.5 \
                --norm_hmap \
                --pred_gamma 2.2 \
                --norm_factor_hmap ${norm_factor} 

export CUDA_VISIBLE_DEVICES=0
img_name=dot_color_close_inv
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
                --BRDF_image_scale 2.5 \
                --norm_hmap \
                --pred_gamma 2.2 \
                --norm_factor_hmap ${norm_factor} 

export CUDA_VISIBLE_DEVICES=0
img_name=dot_color_close
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
                --BRDF_image_scale 2.5 \
                --norm_hmap \
                --pred_gamma 2.2 \
                --norm_factor_hmap ${norm_factor} 