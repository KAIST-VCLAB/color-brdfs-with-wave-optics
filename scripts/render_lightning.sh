noise=0.0
blur_sigma=0.1
date=250929
batchname=lightning

export CUDA_VISIBLE_DEVICES=1
rendername="./output/${date}/${batchname}/range=0.5_sigma=2.6_blur_sigma=0.1_0.1_0.8_lightning_pixelsize=0.8_blue_32/"
pixelsize=0.8
sample_range=0.25
iter=90000

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
                --norm_factor_hmap 0.8 \
                --sample_range ${sample_range} &

export CUDA_VISIBLE_DEVICES=2
rendername="./output/${date}/${batchname}/range=0.5_sigma=2.6_blur_sigma=0.1_0.1_0.8_lightning_pixelsize=0.8_red_32/"

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
                --norm_factor_hmap 0.8 \
                --sample_range ${sample_range} &

export CUDA_VISIBLE_DEVICES=3
rendername="./output/${date}/${batchname}/range=0.5_sigma=2.6_blur_sigma=0.1_0.1_0.8_lightning_pixelsize=0.8_green_32/"

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
                --norm_factor_hmap 0.8 \
                --sample_range ${sample_range} &