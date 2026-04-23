@echo off


:: TRAINING


:: --- Segmentation Model Training ---
echo.
echo [TRAINING] Segmentation Models...


python ./train.py --dataset ISIC-2018 --model UNet --dimension 2d --scaling_version BASIC --epoch 150 --task segmentation || echo [ERROR] UNet failed, starting next


python ./train.py --dataset ISIC-2018 --model PMFSNet --dimension 2d --scaling_version BASIC --epoch 150 --task segmentation || echo [ERROR] PMFSNet failed, starting next


python ./train.py --dataset ISIC-2018 --model EGEUNet --dimension 2d --scaling_version BASIC --epoch 150 --task segmentation || echo [ERROR] EGEUNet failed, starting next


python ./train.py --dataset ISIC-2018 --model LiSANet --dimension 2d --scaling_version BASIC --epoch 150 --task segmentation || echo [ERROR] LiSANet failed, starting next


python ./train.py --dataset ISIC-2018 --model LiSANetMT --dimension 2d --scaling_version BASIC --epoch 150 --task segmentation || echo [ERROR] LiSANetMT failed, starting next



:: --- Classification Model Training ---
echo.
echo [TRAINING] Classification Models...


python ./train.py --dataset ISIC-2018 --model ResNet50 --dimension 2d --scaling_version BASIC --epoch 150 --task classification || echo [ERROR] ResNet50 failed, starting next


python ./train.py --dataset ISIC-2018 --model DenseNet121 --dimension 2d --scaling_version BASIC --epoch 150 --task classification || echo [ERROR] DenseNet121 failed, starting next


python ./train.py --dataset ISIC-2018 --model EfficientNetV2 --dimension 2d --scaling_version BASIC --epoch 150 --task classification || echo [ERROR] EfficientNetV2 failed, starting next


python ./train.py --dataset ISIC-2018 --model MobileNetV3 --dimension 2d --scaling_version BASIC --epoch 150 --task classification || echo [ERROR] MobileNetV3 failed, starting next


python ./train.py --dataset ISIC-2018 --model LiSANetMT --dimension 2d --scaling_version BASIC --epoch 150 --task classification || echo [ERROR] LiSANetMT failed, starting next




@REM :: --- Multitask Model Training ---
@REM echo.
@REM echo [TRAINING] Multitask Models...


@REM python ./train.py --dataset ISIC-2018 --model MBDCNN --dimension 2d --scaling_version BASIC --epoch 150 --task multitask || echo [ERROR] MBDCNN failed, starting next


@REM python ./train.py --dataset ISIC-2018 --model BreastCancerMT --dimension 2d --scaling_version BASIC --epoch 150 --task multitask || echo [ERROR] BreastCancerMT failed, starting next


@REM python ./train.py --dataset ISIC-2018 --model LiSANetMT --dimension 2d --scaling_version BASIC --epoch 150 --task multitask || echo [ERROR] LiSANetMT failed, starting next




@REM :: TESTING


@REM :: --- Segmentation Model Testing ---
@REM echo.
@REM echo [TESTING] Segmentation Models...


@REM python ./test.py --dataset ISIC-2018 --model UNet --pretrain_weight ./pretrain/segmentation/best_UNet.pth --dimension 2d --scaling_version BASIC --task segmentation || echo [ERROR] UNet failed, starting next


@REM python ./test.py --dataset ISIC-2018 --model PMFSNet --pretrain_weight ./pretrain/segmentation/best_PMFSNet.pth --dimension 2d --scaling_version BASIC --task segmentation || echo [ERROR] PMFSNet failed, starting next


@REM python ./test.py --dataset ISIC-2018 --model EGEUNet --pretrain_weight ./pretrain/segmentation/best_EGEUNet.pth --dimension 2d --scaling_version BASIC --task segmentation || echo [ERROR] EGEUNet failed, starting next


@REM python ./test.py --dataset ISIC-2018 --model LiSANet --pretrain_weight ./pretrain/segmentation/best_LiSANet.pth --dimension 2d --scaling_version BASIC --task segmentation || echo [ERROR] LiSANet failed, starting next


@REM python ./test.py --dataset ISIC-2018 --model LiSANetMT --pretrain_weight ./pretrain/segmentation/best_LiSANetMT.pth --dimension 2d --scaling_version BASIC --task segmentation || echo [ERROR] LiSANetMT failed, starting next




@REM :: --- Classification Model Testing ---
@REM echo.
@REM echo [TESTING] Classification Models...


@REM python ./test.py --dataset ISIC-2018 --model ResNet50 --pretrain_weight ./pretrain/classification/best_ResNet50.pth --dimension 2d --scaling_version BASIC --task classification || echo [ERROR] ResNet50 failed, starting next


@REM python ./test.py --dataset ISIC-2018 --model DenseNet121 --pretrain_weight ./pretrain/classification/best_DenseNet121.pth --dimension 2d --scaling_version BASIC --task classification || echo [ERROR] DenseNet121 failed, starting next


@REM python ./test.py --dataset ISIC-2018 --model EfficientNetV2 --pretrain_weight ./pretrain/classification/best_EfficientNetV2.pth --dimension 2d --scaling_version BASIC --task classification || echo [ERROR] EfficientNetV2 failed, starting next


@REM python ./test.py --dataset ISIC-2018 --model MobileNetV3 --pretrain_weight ./pretrain/classification/best_MobileNetV3.pth --dimension 2d --scaling_version BASIC --task classification || echo [ERROR] MobileNetV3 failed, starting next


@REM python ./test.py --dataset ISIC-2018 --model LiSANetMT --pretrain_weight ./pretrain/classification/best_LiSANetMT.pth --dimension 2d --scaling_version BASIC --task classification || echo [ERROR] LiSANetMT failed, starting next




@REM :: --- Multitask Model Testing ---
@REM echo.
@REM echo [TESTING] Multitask Models...


@REM python ./test.py --dataset ISIC-2018 --model MBDCNN --pretrain_weight ./pretrain/multitask/best_MBDCNN.pth --dimension 2d --scaling_version BASIC --task multitask || echo [ERROR] MBDCNN failed, starting next


@REM python ./test.py --dataset ISIC-2018 --model BreastCancerMT --pretrain_weight ./pretrain/multitask/best_BreastCancerMT.pth --dimension 2d --scaling_version BASIC --task multitask || echo [ERROR] BreastCancerMT failed, starting next


@REM python ./test.py --dataset ISIC-2018 --model LiSANetMT --pretrain_weight ./pretrain/multitask/best_LiSANetMT.pth --dimension 2d --scaling_version BASIC --task multitask || echo [ERROR] LiSANetMT failed, starting next


@REM echo.
@REM pause


