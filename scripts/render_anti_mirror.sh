date=251002
batchname=anti

export CUDA_VISIBLE_DEVICES=0
rendername="./output/${date}/${batchname}/sample=1_blur=0.13/0.02_0.8_anti_mirror_pixelsize=1_iri_32/"
pixelsize=1
sample_range=0.3
iter=80000
blur_sigma=0.13
noise=0.0

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

export CUDA_VISIBLE_DEVICES=1
rendername="./output/${date}/${batchname}/sample=1_blur=0.13/0.05_0.3_anti_mirror_pixelsize=0.8_white_32/"
pixelsize=0.8
sample_range=0.3
iter=50000

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
                --sample_range ${sample_range} &

export CUDA_VISIBLE_DEVICES=2
rendername="./output/${date}/${batchname}/sample=1_blur=0.13/0.02_0.4_aniso_anti_mirror_pixelsize=0.8_white_32/"
pixelsize=0.8
sample_range=0.4
iter=40000

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
                --norm_factor_hmap 0.4 \
                --sample_range ${sample_range} &

export CUDA_VISIBLE_DEVICES=3
rendername="./output/${date}/${batchname}/sample=1_blur=0.13/0.02_0.8_aniso_anti_mirror_pixelsize=0.5_iri_32/"
pixelsize=0.5
sample_range=0.4
iter=40000

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
                
export CUDA_VISIBLE_DEVICES=4
rendername="./output/${date}/${batchname}/sample=1_blur=0.13/0.02_0.8_aniso_anti_mirror_pixelsize=0.5_red_32/"
pixelsize=0.5
sample_range=0.4
iter=40000

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
                             
export CUDA_VISIBLE_DEVICES=5
rendername="./output/${date}/${batchname}/sample=1_blur=0.13/0.02_0.8_anti_mirror_pixelsize=0.5_red_32/"
pixelsize=0.5
sample_range=0.3
iter=40000

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