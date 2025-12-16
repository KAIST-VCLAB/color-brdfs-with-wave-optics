noise=0.0
blur_sigma=0.1
date=250930
batchname=pic_inv

export CUDA_VISIBLE_DEVICES=0
rendername="./output/${date}/${batchname}/sigma=2.6_blur_sigma=0.1___butterfly_pixelsize=0.8_white_32/"
pixelsize=0.8
sample_range=0.3
iter=70000

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
                --sample_range ${sample_range} &

export CUDA_VISIBLE_DEVICES=1
rendername="./output/${date}/${batchname}/sigma=2.6_blur_sigma=0.1___lightning_pixelsize=0.8_white_32/"

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
                --sample_range ${sample_range} &
                
export CUDA_VISIBLE_DEVICES=2
rendername="./output/${date}/${batchname}/sigma=2.6_blur_sigma=0.1___rabbit_pixelsize=0.8_white_32/"

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
                --sample_range ${sample_range} &
                
export CUDA_VISIBLE_DEVICES=3
rendername="./output/${date}/${batchname}/sigma=2.6_blur_sigma=0.1___heart_pixelsize=0.8_white_32/"

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
                --sample_range ${sample_range} &
                
export CUDA_VISIBLE_DEVICES=4
rendername="./output/${date}/${batchname}/sigma=2.6_blur_sigma=0.1___note_pixelsize=0.8_white_32/"

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
                --sample_range ${sample_range} &