definter="bilin"
INTERPOLATION=${1:-$definter}

med_names=('validation/med_res/abc_0050_00500348_fae0ecd8b3dc068d39f0d09c_000' \
'validation/med_res/abc_0050_00500149_54930d6f7740b03347d89a56_000' \
'validation/med_res/abc_0050_00500166_5894bbd701b2bb0fc88a6978_007' \
'validation/med_res/abc_0050_00500082_4cb4bf14428fe3832dd7ed78_000' \
'validation/med_res/abc_0050_00500683_511f7debb63f164003339dec_000')

for str in ${med_names[*]}; do
    python3 scripts/fuse_images.py \
        --true-filename ${str}.hdf5 \
        --pred-filename ${str}__predictions.hdf5 \
        --output-dir out/med/${INTERPOLATION} \
        -i ${INTERPOLATION} \
        -r 0.05
done

high_names=('validation/high_res/abc_0050_00500348_fae0ecd8b3dc068d39f0d09c_000' \
'/validation/high_res/abc_0050_00500166_5894bbd701b2bb0fc88a6978_007')

for str in ${high_names[*]}; do
    python3 scripts/fuse_images.py \
        --true-filename validation/high_res/${str}.hdf5 \
        --pred-filename validation/high_res/${str}__predictions.hdf5 \
        --output-dir out/high/${INTERPOLATION} \
        -i ${INTERPOLATION} \
        -r 0.02
done

# med_names=('validation/med_res/abc_0050_00500348_fae0ecd8b3dc068d39f0d09c_000' \
# 'validation/med_res/abc_0050_00500149_54930d6f7740b03347d89a56_000')

# for str in ${med_names[*]}; do
#     python3 scripts/fuse_images.py \
#         --true-filename ${str}.hdf5 \
#         --pred-filename ${str}__predictions.hdf5 \
#         --output-dir out/med/${INTERPOLATION} \
#         -i bisplain \
#         -r 0.05
# done