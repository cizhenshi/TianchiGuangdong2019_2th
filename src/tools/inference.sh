GPUS=$1
./tools/dist_test.sh ./configs/tianchi/faster_rcnn_ohem_r50_fpn_test.py  \
          ./work_dirs/faster/faster_res50.pth $GPUS --out ../data/faster.pkl

./tools/dist_test.sh ./configs/tianchi/cascade_resnext101.py  \
          ./work_dirs/casres101/epoch_45.pth $GPUS --out ../data/cas101.pkl

python ./tools/merge.py
