noise=0.0
blur_sigma=0.5
date=250930
batchname=pic

export CUDA_VISIBLE_DEVICES=0
rendername="./output/${date}/${batchname}/sigma=3.9_blur_sigma=0.5_0.006_0.8_note_pixelsize=2_white_32/"
pixelsize=2
sample_range=0.3
iter=100000

python render_fab.py --resolution 32 \
                --loadmodel ${rendername}model/ckpt${iter}.tsr \
                --model_path ${rendername} \
                --render_scale 1 \
                --C1 Kirchhoff \
                --C3 nonparaxial \
                --material "Aluminium" \
                --noise ${noise} \
                --blur_sigma ${blur_sigma} \
                --sigma 3.9 \
                --pixelsize ${pixelsize} \
                --norm_hmap \
                --norm_factor_hmap 0.8 \
                --sample_range ${sample_range} &

export CUDA_VISIBLE_DEVICES=1
rendername="./output/${date}/${batchname}/sigma=3.9_blur_sigma=0.5_0.01_0.8_lightning_pixelsize=2_white_32/"

python render_fab.py --resolution 32 \
                --loadmodel ${rendername}model/ckpt${iter}.tsr \
                --model_path ${rendername} \
                --render_scale 1 \
                --C1 Kirchhoff \
                --C3 nonparaxial \
                --material "Aluminium" \
                --noise ${noise} \
                --blur_sigma ${blur_sigma} \
                --sigma 3.9 \
                --pixelsize ${pixelsize} \
                --norm_hmap \
                --norm_factor_hmap 0.8 \
                --sample_range ${sample_range} &
                
export CUDA_VISIBLE_DEVICES=2
rendername="./output/${date}/${batchname}/sigma=3.9_blur_sigma=0.5_0.04_0.8_butterfly_pixelsize=2_white_32/"

python render_fab.py --resolution 32 \
                --loadmodel ${rendername}model/ckpt${iter}.tsr \
                --model_path ${rendername} \
                --render_scale 1 \
                --C1 Kirchhoff \
                --C3 nonparaxial \
                --material "Aluminium" \
                --noise ${noise} \
                --blur_sigma ${blur_sigma} \
                --sigma 3.9 \
                --pixelsize ${pixelsize} \
                --norm_hmap \
                --norm_factor_hmap 0.8 \
                --sample_range ${sample_range} &
                
export CUDA_VISIBLE_DEVICES=3
rendername="./output/${date}/${batchname}/sigma=3.9_blur_sigma=0.5_0.04_0.8_heart_pixelsize=2_white_32/"

python render_fab.py --resolution 32 \
                --loadmodel ${rendername}model/ckpt${iter}.tsr \
                --model_path ${rendername} \
                --render_scale 1 \
                --C1 Kirchhoff \
                --C3 nonparaxial \
                --material "Aluminium" \
                --noise ${noise} \
                --blur_sigma ${blur_sigma} \
                --sigma 3.9 \
                --pixelsize ${pixelsize} \
                --norm_hmap \
                --norm_factor_hmap 0.8 \
                --sample_range ${sample_range} &
                
export CUDA_VISIBLE_DEVICES=4
rendername="./output/${date}/${batchname}/sigma=3.9_blur_sigma=0.5_0.04_0.8_rabbit_pixelsize=2_white_32/"

python render_fab.py --resolution 32 \
                --loadmodel ${rendername}model/ckpt${iter}.tsr \
                --model_path ${rendername} \
                --render_scale 1 \
                --C1 Kirchhoff \
                --C3 nonparaxial \
                --material "Aluminium" \
                --noise ${noise} \
                --blur_sigma ${blur_sigma} \
                --sigma 3.9 \
                --pixelsize ${pixelsize} \
                --norm_hmap \
                --norm_factor_hmap 0.8 \
                --sample_range ${sample_range}