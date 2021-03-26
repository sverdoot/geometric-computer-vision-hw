definter="bilin"
INTERPOLATION=${1:-$definter}

med_names=("abc_0050_00500166_5894bbd701b2bb0fc88a6978_007")

for str in ${med_names[*]}; do
    python3 scripts/fuse_images.py \
        --true-filename validation/med_res/${str}.hdf5 \
        --pred-filename validation/med_res/${str}__predictions.hdf5 \
        --output-dir out/med/${INTERPOLATION} \
        -i ${INTERPOLATION}
done

high_names=() #"abc_0050_00500082_4cb4bf14428fe3832dd7ed78_000")

for str in ${high_names[*]}; do
    python3 scripts/fuse_images.py \
        --true-filename validation/high_res/${str}.hdf5 \
        --pred-filename validation/high_res/${str}__predictions.hdf5 \
        --output-dir out/high/${INTERPOLATION} \
        -i ${INTERPOLATION}
done