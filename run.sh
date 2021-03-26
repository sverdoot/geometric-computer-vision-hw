# RES_MED=0.05
# RES_HIGH=0.02

# med

python3 scripts/fuse_images.py -t=validation/med_res/abc_0050_00500082_4cb4bf14428fe3832dd7ed78_000.hdf5 -p=validation/med_res/abc_0050_00500082_4cb4bf14428fe3832dd7ed78_000__predictions.hdf5 -o=out

# python3 scripts/fuse_images.py -t=validation/med_res/abc_0050_00500149_54930d6f7740b03347d89a56_000.hdf5 -p=validation/med_res/abc_0050_00500149_54930d6f7740b03347d89a56_000__predictions.hdf5 -o=results --resolution_3d=$RES_MED

# python3 scripts/fuse_images.py -t=validation/med_res/abc_0050_00500166_5894bbd701b2bb0fc88a6978_007.hdf5 -p=validation/med_res/abc_0050_00500166_5894bbd701b2bb0fc88a6978_007__predictions.hdf5 -o=results --resolution_3d=$RES_MED

# python3 scripts/fuse_images.py -t=validation/med_res/abc_0050_00500348_fae0ecd8b3dc068d39f0d09c_000.hdf5 -p=validation/med_res/abc_0050_00500348_fae0ecd8b3dc068d39f0d09c_000__predictions.hdf5 -o=results --resolution_3d=$RES_MED

# python3 scripts/fuse_images.py -t=validation/med_res/abc_0050_00500683_511f7debb63f164003339dec_000.hdf5 -p=validation/med_res/abc_0050_00500683_511f7debb63f164003339dec_000__predictions.hdf5 -o=results --resolution_3d=$RES_MED


# high

# python3 scripts/fuse_images.py -t=validation/high_res/abc_0050_00500166_5894bbd701b2bb0fc88a6978_007.hdf5 -p=validation/high_res/abc_0050_00500166_5894bbd701b2bb0fc88a6978_007__predictions.hdf5 -o=results_high --resolution_3d=$RES_HIGH

# python3 scripts/fuse_images.py -t=validation/high_res/abc_0050_00500348_fae0ecd8b3dc068d39f0d09c_000.hdf5 -p=validation/high_res/abc_0050_00500348_fae0ecd8b3dc068d39f0d09c_000__predictions.hdf5 -o=results_high --resolution_3d=$RES_HIGH
