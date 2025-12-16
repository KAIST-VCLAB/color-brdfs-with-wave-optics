noise=0.0
blur_sigma=0.13
date=250928
batchname=uniform

export CUDA_VISIBLE_DEVICES=0
rendername="./output/${date}/${batchname}/sigma=2.6_blur_sigma=0.13_0.16_1.2_uniform_pixelsize=0.8_blue_32/"
pixelsize=0.8
sample_range=0.25
iter=30000

python render_fab.py --resolution 32 \
                --loadmodel ${rendername}model/ckpt${iter}.tsr \
                --model_path ${rendername} \
                --render_scale 1 \
                --C1 Kirchhoff \
                --C3 nonparaxial \
                --material "Aluminium" \
                --noise ${noise} \
                --blur_sigma ${blur_sigma} \
                --sigma 2.6 \
                --pixelsize ${pixelsize} \
                --norm_hmap \
                --norm_factor_hmap 1.2 \
                --sample_range ${sample_range} &

export CUDA_VISIBLE_DEVICES=1
rendername="./output/${date}/${batchname}/sigma=2.6_blur_sigma=0.13_0.16_1.2_uniform_pixelsize=0.8_red_32/"
pixelsize=0.8
sample_range=0.25

python render_fab.py --resolution 32 \
                --loadmodel ${rendername}model/ckpt${iter}.tsr \
                --model_path ${rendername} \
                --render_scale 1 \
                --C1 Kirchhoff \
                --C3 nonparaxial \
                --material "Aluminium" \
                --noise ${noise} \
                --blur_sigma ${blur_sigma} \
                --sigma 2.6 \
                --pixelsize ${pixelsize} \
                --norm_hmap \
                --norm_factor_hmap 1.2 \
                --sample_range ${sample_range} &

export CUDA_VISIBLE_DEVICES=2
rendername="./output/${date}/${batchname}/sigma=2.6_blur_sigma=0.13_0.16_1.2_uniform_pixelsize=0.8_green_32/"
pixelsize=0.8
sample_range=0.25

python render_fab.py --resolution 32 \
                --loadmodel ${rendername}model/ckpt${iter}.tsr \
                --model_path ${rendername} \
                --render_scale 1 \
                --C1 Kirchhoff \
                --C3 nonparaxial \
                --material "Aluminium" \
                --noise ${noise} \
                --blur_sigma ${blur_sigma} \
                --sigma 2.6 \
                --pixelsize ${pixelsize} \
                --norm_hmap \
                --norm_factor_hmap 1.2 \
                --sample_range ${sample_range}  &

export CUDA_VISIBLE_DEVICES=3
rendername="./output/${date}/${batchname}/sigma=2.6_blur_sigma=0.13_0.16_1.2_uniform_pixelsize=0.8_cyan_32/"
pixelsize=0.8
sample_range=0.25

python render_fab.py --resolution 32 \
                --loadmodel ${rendername}model/ckpt${iter}.tsr \
                --model_path ${rendername} \
                --render_scale 1 \
                --C1 Kirchhoff \
                --C3 nonparaxial \
                --material "Aluminium" \
                --noise ${noise} \
                --blur_sigma ${blur_sigma} \
                --sigma 2.6 \
                --pixelsize ${pixelsize} \
                --norm_hmap \
                --norm_factor_hmap 1.2 \
                --sample_range ${sample_range} &

export CUDA_VISIBLE_DEVICES=4
rendername="./output/${date}/${batchname}/sigma=2.6_blur_sigma=0.13_0.16_1.2_uniform_pixelsize=0.8_purple_32/"
pixelsize=0.8
sample_range=0.25

python render_fab.py --resolution 32 \
                --loadmodel ${rendername}model/ckpt${iter}.tsr \
                --model_path ${rendername} \
                --render_scale 1 \
                --C1 Kirchhoff \
                --C3 nonparaxial \
                --material "Aluminium" \
                --noise ${noise} \
                --blur_sigma ${blur_sigma} \
                --sigma 2.6 \
                --pixelsize ${pixelsize} \
                --norm_hmap \
                --norm_factor_hmap 1.2 \
                --sample_range ${sample_range}  &

export CUDA_VISIBLE_DEVICES=5
rendername="./output/${date}/${batchname}/sigma=2.6_blur_sigma=0.13_0.16_1.2_uniform_pixelsize=0.8_yellow_32/"
pixelsize=0.8
sample_range=0.25

python render_fab.py --resolution 32 \
                --loadmodel ${rendername}model/ckpt${iter}.tsr \
                --model_path ${rendername} \
                --render_scale 1 \
                --C1 Kirchhoff \
                --C3 nonparaxial \
                --material "Aluminium" \
                --noise ${noise} \
                --blur_sigma ${blur_sigma} \
                --sigma 2.6 \
                --pixelsize ${pixelsize} \
                --norm_hmap \
                --norm_factor_hmap 1.2 \
                --sample_range ${sample_range}  &

export CUDA_VISIBLE_DEVICES=6
rendername="./output/${date}/${batchname}/sigma=2.6_blur_sigma=0.13_0.16_1.2_uniform_pixelsize=0.8_orange_32/"
pixelsize=0.8
sample_range=0.25

python render_fab.py --resolution 32 \
                --loadmodel ${rendername}model/ckpt${iter}.tsr \
                --model_path ${rendername} \
                --render_scale 1 \
                --C1 Kirchhoff \
                --C3 nonparaxial \
                --material "Aluminium" \
                --noise ${noise} \
                --blur_sigma ${blur_sigma} \
                --sigma 2.6 \
                --pixelsize ${pixelsize} \
                --norm_hmap \
                --norm_factor_hmap 1.2 \
                --sample_range ${sample_range} 