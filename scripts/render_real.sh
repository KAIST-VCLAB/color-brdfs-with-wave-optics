noise=0.0
date=250927
batchname=real

export CUDA_VISIBLE_DEVICES=0
rendername="./output/${date}/${batchname}/sigma=2.6_blur_sigma=0.5_0.006_0.8_lightning_pixelsize=1.5_white_32/"
pixelsize=1.5
blur_sigma=0.5
sample_range=0.3
iter=170000

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
                --sample_range ${sample_range} 

rendername="./output/${date}/${batchname}/sigma=2.6_blur_sigma=0.65_0.006_0.8_aniso_anti_mirror_pixelsize=2_white_32/"
pixelsize=2
blur_sigma=0.65
sample_range=0.4
iter=190000

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
                --sample_range ${sample_range} 
                
rendername="./output/${date}/${batchname}/sigma=2.6_blur_sigma=0.65_0.01_0.8_anti_mirror_pixelsize=2_white_32/"
pixelsize=2
blur_sigma=0.65
sample_range=0.3
iter=230000

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
                --sample_range ${sample_range}

rendername="./output/${date}/${batchname}/sigma=3.9_blur_sigma=0.65_0.01_0.8_note_pixelsize=2_white_32/"
pixelsize=2
blur_sigma=0.65
sample_range=0.3
iter=150000

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

rendername="./output/${date}/${batchname}/sigma=5.2_blur_sigma=0.65_0.01_0.8_inverse_batman_pixelsize=2_white_32/"
pixelsize=2
blur_sigma=0.65
sample_range=0.3
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
                --sigma 5.2 \
                --pixelsize ${pixelsize} \
                --norm_hmap \
                --norm_factor_hmap 0.8 \
                --sample_range ${sample_range}

rendername="./output/${date}/${batchname}/sigma=5.2_blur_sigma=0.65_0.02_0.8_inverse_batman_pixelsize=2_white_32/"
pixelsize=2
blur_sigma=0.65
sample_range=0.3
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
                --sigma 5.2 \
                --pixelsize ${pixelsize} \
                --norm_hmap \
                --norm_factor_hmap 0.8 \
                --sample_range ${sample_range} 

rendername="./output/${date}/${batchname}/sigma=2.6_blur_sigma=0.65_0.03_0.8_diffuse_pixelsize=2_disk_iri_32/"
pixelsize=2
blur_sigma=0.65
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
                --sample_range ${sample_range}


rendername="./output/${date}/${batchname}/sigma=2.6_blur_sigma=0.65_0.03_0.8_diffuse_pixelsize=2_inv_disk_iri_32/"
pixelsize=2
blur_sigma=0.65
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
                --sample_range ${sample_range} 

rendername="./output/${date}/${batchname}/sigma=3.9_blur_sigma=0.65_0.006_0.8_dot_color_close_inv_pixelsize=1.5_white_32/"
pixelsize=1.5
blur_sigma=0.65
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
                --sample_range ${sample_range} 
                
rendername="./output/${date}/${batchname}/sigma=3.9_blur_sigma=0.65_0.006_0.8_dot_color_close_pixelsize=1.5_white_32/"
pixelsize=1.5
blur_sigma=0.65
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
                --sample_range ${sample_range} 
                
rendername="./output/${date}/${batchname}/sigma=3.9_blur_sigma=0.65_0.006_0.8_dots_color_close_pixelsize=1.5_white_32/"
pixelsize=1.5
blur_sigma=0.65
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
                --sample_range ${sample_range} 
                      
rendername="./output/${date}/${batchname}/sigma=3.9_blur_sigma=0.65_0.006_0.8_dots_color_close_inv_pixelsize=1.5_white_32/"
pixelsize=1.5
blur_sigma=0.65
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
                --sample_range ${sample_range} 