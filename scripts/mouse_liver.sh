python run.py \
    --gene_image datasets/mouse_liver/C01344C4.tsv.gz \
    --stain_image datasets/mouse_liver/C01344C4.tif \
    --out_dir datasets/mouse_liver/results/ \
    --max_size 512 \
    --loss MSE \
    --lr 0.01 \
    --niter 1000 \
    --optimizer Adam \
    --flip_h 0 \
    --flip_v 1 \
    --rot90 3

