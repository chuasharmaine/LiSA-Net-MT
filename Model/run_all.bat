@echo off

echo ===== STARTING ALL EXPERIMENTS =====

REM =========================
REM CLASSIFICATION
REM =========================
echo Running ResNet-50 (Classification)
python ./train.py --dataset ISIC-2018 --model ResNet50 --dimension 2d --scaling_version BASIC --epoch 150 --task classification
if errorlevel 1 echo  EfficientNetV2 FAILED, moving on...

echo Running DenseNet-121 (Classification)
python ./train.py --dataset ISIC-2018 --model DenseNet121 --dimension 2d --scaling_version BASIC --epoch 150 --task classification
if errorlevel 1 echo  EfficientNetV2 FAILED, moving on...

echo Running EfficientNetV2 (Classification)
python ./train.py --dataset ISIC-2018 --model EfficientNetV2 --dimension 2d --scaling_version BASIC --epoch 150 --task classification
if errorlevel 1 echo  EfficientNetV2 FAILED, moving on...

echo Running MobileNetV3 (Classification)
python ./train.py --dataset ISIC-2018 --model MobileNetV3 --dimension 2d --scaling_version BASIC --epoch 150 --task classification
if errorlevel 1 echo  MobileNetV3 FAILED, moving on...

echo Running LiSA-Net-MT (Classification)
python ./train.py --dataset ISIC-2018 --model LiSANetMT --dimension 2d --scaling_version BASIC --epoch 150 --task classification
if errorlevel 1 echo  LiSA-Net-MT (Classification) FAILED, moving on...


REM =========================
REM MULTITASK
REM =========================
echo Running MBDCNN (Multitask)
python ./train.py --dataset ISIC-2018 --model MBDCNN --dimension 2d --scaling_version BASIC --epoch 150 --task multitask
if errorlevel 1 echo  MBDCNN FAILED, moving on...

echo Running BreastCancerMT (Multitask)
python ./train.py --dataset ISIC-2018 --model BreastCancerMT --dimension 2d --scaling_version BASIC --epoch 150 --task multitask
if errorlevel 1 echo  BreastCancerMT FAILED, moving on...

echo Running LiSA-Net-MT (Multitask)
python ./train.py --dataset ISIC-2018 --model LiSANetMT --dimension 2d --scaling_version BASIC --epoch 150 --task multitask
if errorlevel 1 echo  LiSA-Net-MT (Multitask) FAILED, moving on...

echo ===== ALL EXPERIMENTS FINISHED (with possible errors) =====
pause
