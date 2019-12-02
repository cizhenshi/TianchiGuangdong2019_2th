cd ./data
mkdir ./round2_data
mkdir ./round2_data/Annotations
mkdir ./round2_data/defect
mkdir ./round2_data/normal
mv ./guangdong1_round2_train2_20191004_Annotations/Annotations/anno_train.json \
./round2_data/Annotations/anno_train_1004.json
mv ./guangdong1_round2_train_part1_20190924/Annotations/anno_train.json ./round2_data/Annotations/anno_train_0924.json
mv ./guangdong1_round2_train_part1_20190924/defect/* ./round2_data/defect/
mv ./guangdong1_round2_train_part2_20190924/defect/* ./round2_data/defect/
mv ./guangdong1_round2_train_part3_20190924/defect/* ./round2_data/defect/
mv ./guangdong1_round2_train_part2_20190924/normal/* ./round2_data/normal/
mv ./guangdong1_round2_train2_20191004_images/defect/* ./round2_data/defect/
mv ./guangdong1_round2_train2_20191004_images/normal/* ./round2_data/normal/
cd ../src
python tools/PrepareData.py