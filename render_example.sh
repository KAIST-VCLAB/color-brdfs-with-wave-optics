date=250926
batchname=train_example

export CUDA_VISIBLE_DEVICES=0
rendername="./output/${date}/${batchname}/sample=1_blur=0.13/0.05_0.3_anti_mirror_pixelsize=0.8_white/"
pixelsize=0.8
sample_range=0.3
iter=60000
noise=0.0
blur_sigma=0.13

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
                --norm_factor_hmap 0.3 \
                --sample_range ${sample_range} 