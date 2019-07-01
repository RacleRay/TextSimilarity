python graph_feature_generate.py
python pretrain.py -train valid=0
python train.py -train valid=0
python pretrain.py -predict
cp bestmodel.h5 bestmodel0.h5
cp result.csv result0.csv
rm bestmodel.h5

python build_ext_data.py
python pretrain.py -train valid=1
python train.py -train valid=1
python pretrain.py -predict
cp bestmodel.h5 bestmodel1.h5
cp result.csv result1.csv
rm bestmodel.h5

python build_ext_data.py
python pretrain.py -train valid=2
python train.py -train valid=2
python pretrain.py -predict
cp bestmodel.h5 bestmodel2.h5
cp result.csv result2.csv
rm bestmodel.h5

python build_ext_data.py
python pretrain.py -train valid=3
python train.py -train valid=3
python pretrain.py -predict
cp bestmodel.h5 bestmodel3.h5
cp result.csv result3.csv
rm bestmodel.h5

python build_ext_data.py
python pretrain.py -train valid=4
python train.py -train valid=4
python pretrain.py -predict
cp bestmodel.h5 bestmodel4.h5
cp result.csv result4.csv
rm bestmodel.h5

python build_ext_data.py
python pretrain.py -train valid=5
python train.py -train valid=5
python pretrain.py -predict
cp bestmodel.h5 bestmodel5.h5
cp result.csv result5.csv
rm bestmodel.h5

python build_ext_data.py
python pretrain.py -train valid=6
python train.py -train valid=6
python pretrain.py -predict
cp bestmodel.h5 bestmodel6.h5
cp result.csv result6.csv
rm bestmodel.h5

python build_ext_data.py
python pretrain.py -train valid=7
python train.py -train valid=7
python pretrain.py -predict
cp bestmodel.h5 bestmodel7.h5
cp result.csv result7.csv
rm bestmodel.h5

python build_ext_data.py
python pretrain.py -train valid=8
python train.py -train valid=8
python pretrain.py -predict
cp bestmodel.h5 bestmodel8.h5
cp result.csv result8.csv
rm bestmodel.h5

python build_ext_data.py
python pretrain.py -train valid=9
python train.py -train valid=9
python pretrain.py -predict
cp bestmodel.h5 bestmodel9.h5
cp result.csv result9.csv
rm bestmodel.h5

rm result.csv
python avg.py
