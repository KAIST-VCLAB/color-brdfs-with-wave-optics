noise=0.0
blur_sigma=0.1
date=251001
batchname=circle

export CUDA_VISIBLE_DEVICES=4
rendername="./output/${date}/${batchname}/iri_range=0.09_sigma=0.1_0.02_0.8_diffuse_pixelsize=0.8_inv_disk_iri_32/"
pixelsize=0.8
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
                --sigma 2.6 \
                --pixelsize ${pixelsize} \
                --norm_hmap \
                --norm_factor_hmap 0.8 \
                --sample_range ${sample_range} &

export CUDA_VISIBLE_DEVICES=5
rendername="./output/${date}/${batchname}/iri_range=0.09_sigma=0.1_0.02_0.8_diffuse_pixelsize=0.8_disk_iri_32/"

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

