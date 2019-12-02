path=$1
python ./tools/inference.py --config ../config/cascade_x101_20e.py \
        --checkpoint ../weights/x101.pth \
        --out ./x101.json \
        --path $path
python ./tools/inference.py --config ../config/cascade_r101.py \
        --checkpoint ../weights/r101.pth \
        --out ./r101.json \
        --path $path
python ./tools/model_ensemble.py
