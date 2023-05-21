[05/08 09:14:27] fastreid INFO: Rank of current process: 0. World size: 1
[05/08 09:14:28] fastreid INFO: Environment info:
----------------------  ----------------------------------------------------------------------------------------------------
sys.platform            win32
Python                  3.9.16 (main, Mar  8 2023, 10:39:24) [MSC v.1916 64 bit (AMD64)]
numpy                   1.23.5
fastreid                1.3 @C:\source\GitRepo\reid\fast-reid\.\fastreid
FASTREID_ENV_MODULE     <not set>
PyTorch                 1.13.0 @C:\Users\wsyconan\.conda\envs\fastreid_2\lib\site-packages\torch
PyTorch debug build     False
GPU available           True
GPU 0                   NVIDIA GeForce GTX 1050 Ti
CUDA_HOME               C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7
Pillow                  9.4.0
torchvision             0.14.0+cpu @C:\Users\wsyconan\.conda\envs\fastreid_2\lib\site-packages\torchvision
torchvision arch flags  C:\Users\wsyconan\.conda\envs\fastreid_2\lib\site-packages\torchvision\_C.pyd; cannot find cuobjdump
cv2                     4.7.0
----------------------  ----------------------------------------------------------------------------------------------------
PyTorch built with:
  - C++ Version: 199711
  - MSVC 192829337
  - Intel(R) Math Kernel Library Version 2020.0.2 Product Build 20200624 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v2.6.0 (Git Hash 52b5f107dd9cf10910aaa19cb47f3abf9b349815)
  - OpenMP 2019
  - LAPACK is enabled (usually provided by MKL)
  - CPU capability usage: AVX2
  - CUDA Runtime 11.7
  - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_61,code=sm_61;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_37,code=compute_37
  - CuDNN 8.5
  - Magma 2.5.4
  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CUDA_VERSION=11.7, CUDNN_VERSION=8.5.0, CXX_COMPILER=C:/cb/pytorch_1000000000000/work/tmp_bin/sccache-cl.exe, CXX_FLAGS=/DWIN32 /D_WINDOWS /GR /EHsc /w /bigobj -DUSE_PTHREADPOOL -openmp:experimental -IC:/cb/pytorch_1000000000000/work/mkl/include -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DUSE_FBGEMM -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -DEDGE_PROFILER_USE_KINETO, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.13.0, USE_CUDA=ON, USE_CUDNN=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=OFF, USE_OPENMP=ON, USE_ROCM=OFF, 

[05/08 09:14:28] fastreid INFO: Command line arguments: Namespace(config_file='./configs/MSMT17/bagtricks_R50-ibn.yml', resume=False, eval_only=True, num_gpus=1, num_machines=1, machine_rank=0, dist_url='tcp://127.0.0.1:49153', opts=['MODEL.WEIGHTS', 'checkpoints/msmt_bot_R50-ibn.pth', 'MODEL.DEVICE', 'cuda:0'])
[05/08 09:14:28] fastreid INFO: Contents of args.config_file=./configs/MSMT17/bagtricks_R50-ibn.yml:
_BASE_: ../Base-bagtricks.yml

MODEL:
  BACKBONE:
    WITH_IBN: True

DATASETS:
  NAMES: ("MSMT17",)
  TESTS: ("MSMT17",)

OUTPUT_DIR: logs/msmt17/bagtricks_R50-ibn


[05/08 09:14:28] fastreid INFO: Running with full config:
CUDNN_BENCHMARK: True
DATALOADER:
  NUM_INSTANCE: 4
  NUM_WORKERS: 8
  SAMPLER_TRAIN: NaiveIdentitySampler
  SET_WEIGHT: []
DATASETS:
  COMBINEALL: False
  NAMES: ('MSMT17',)
  TESTS: ('MSMT17',)
INPUT:
  AFFINE:
    ENABLED: False
  AUGMIX:
    ENABLED: False
    PROB: 0.0
  AUTOAUG:
    ENABLED: False
    PROB: 0.0
  CJ:
    BRIGHTNESS: 0.15
    CONTRAST: 0.15
    ENABLED: False
    HUE: 0.1
    PROB: 0.5
    SATURATION: 0.1
  CROP:
    ENABLED: False
    RATIO: [0.75, 1.3333333333333333]
    SCALE: [0.16, 1]
    SIZE: [224, 224]
  FLIP:
    ENABLED: True
    PROB: 0.5
  PADDING:
    ENABLED: True
    MODE: constant
    SIZE: 10
  REA:
    ENABLED: True
    PROB: 0.5
    VALUE: [123.675, 116.28, 103.53]
  RPT:
    ENABLED: False
    PROB: 0.5
  SIZE_TEST: [256, 128]
  SIZE_TRAIN: [256, 128]
KD:
  EMA:
    ENABLED: False
    MOMENTUM: 0.999
  MODEL_CONFIG: []
  MODEL_WEIGHTS: []
MODEL:
  BACKBONE:
    ATT_DROP_RATE: 0.0
    DEPTH: 50x
    DROP_PATH_RATIO: 0.1
    DROP_RATIO: 0.0
    FEAT_DIM: 2048
    LAST_STRIDE: 1
    NAME: build_resnet_backbone
    NORM: BN
    PRETRAIN: True
    PRETRAIN_PATH: 
    SIE_COE: 3.0
    STRIDE_SIZE: (16, 16)
    WITH_IBN: True
    WITH_NL: False
    WITH_SE: False
  DEVICE: cuda:0
  FREEZE_LAYERS: []
  HEADS:
    CLS_LAYER: Linear
    EMBEDDING_DIM: 0
    MARGIN: 0.0
    NAME: EmbeddingHead
    NECK_FEAT: before
    NORM: BN
    NUM_CLASSES: 0
    POOL_LAYER: GlobalAvgPool
    SCALE: 1
    WITH_BNNECK: True
  LOSSES:
    CE:
      ALPHA: 0.2
      EPSILON: 0.1
      SCALE: 1.0
    CIRCLE:
      GAMMA: 128
      MARGIN: 0.25
      SCALE: 1.0
    COSFACE:
      GAMMA: 128
      MARGIN: 0.25
      SCALE: 1.0
    FL:
      ALPHA: 0.25
      GAMMA: 2
      SCALE: 1.0
    NAME: ('CrossEntropyLoss', 'TripletLoss')
    TRI:
      HARD_MINING: True
      MARGIN: 0.3
      NORM_FEAT: False
      SCALE: 1.0
  META_ARCHITECTURE: Baseline
  PIXEL_MEAN: [123.675, 116.28, 103.53]
  PIXEL_STD: [58.395, 57.120000000000005, 57.375]
  QUEUE_SIZE: 8192
  WEIGHTS: checkpoints/msmt_bot_R50-ibn.pth
OUTPUT_DIR: logs/msmt17/bagtricks_R50-ibn
SOLVER:
  AMP:
    ENABLED: True
  BASE_LR: 0.00035
  BIAS_LR_FACTOR: 1.0
  CHECKPOINT_PERIOD: 30
  CLIP_GRADIENTS:
    CLIP_TYPE: norm
    CLIP_VALUE: 5.0
    ENABLED: False
    NORM_TYPE: 2.0
  DELAY_EPOCHS: 0
  ETA_MIN_LR: 1e-07
  FREEZE_ITERS: 0
  GAMMA: 0.1
  HEADS_LR_FACTOR: 1.0
  IMS_PER_BATCH: 64
  MAX_EPOCH: 120
  MOMENTUM: 0.9
  NESTEROV: False
  OPT: Adam
  SCHED: MultiStepLR
  STEPS: [40, 90]
  WARMUP_FACTOR: 0.1
  WARMUP_ITERS: 2000
  WARMUP_METHOD: linear
  WEIGHT_DECAY: 0.0005
  WEIGHT_DECAY_BIAS: 0.0005
  WEIGHT_DECAY_NORM: 0.0005
TEST:
  AQE:
    ALPHA: 3.0
    ENABLED: False
    QE_K: 5
    QE_TIME: 1
  EVAL_PERIOD: 30
  FLIP:
    ENABLED: False
  IMS_PER_BATCH: 128
  METRIC: cosine
  PRECISE_BN:
    DATASET: Market1501
    ENABLED: False
    NUM_ITER: 300
  RERANK:
    ENABLED: False
    K1: 20
    K2: 6
    LAMBDA: 0.3
  ROC:
    ENABLED: False
[05/08 09:14:28] fastreid INFO: Full config saved to C:\source\GitRepo\reid\fast-reid\logs\msmt17\bagtricks_R50-ibn\config.yaml
[05/08 09:14:28] fastreid.utils.env INFO: Using a generated random seed 28814457
[05/08 09:14:29] fastreid.engine.defaults INFO: Model:
Baseline(
  (backbone): ResNet(
    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (bn1): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
    (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
    (layer1): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
        (downsample): Sequential(
          (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (2): Bottleneck(
        (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
    )
    (layer2): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn2): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
        (downsample): Sequential(
          (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (2): Bottleneck(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (3): Bottleneck(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
    )
    (layer3): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn2): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
        (downsample): Sequential(
          (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (2): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (3): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (4): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (5): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
    )
    (layer4): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
        (downsample): Sequential(
          (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (2): Bottleneck(
        (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
    )
  )
  (heads): EmbeddingHead(
    (pool_layer): GlobalAvgPool(output_size=1)
    (bottleneck): Sequential(
      (0): BatchNorm(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (cls_layer): Linear(num_classes=0, scale=1, margin=0.0)
  )
)
[05/08 09:14:29] fastreid.utils.checkpoint INFO: Loading checkpoint from checkpoints/msmt_bot_R50-ibn.pth
[05/08 09:14:29] fastreid.utils.checkpoint INFO: Some model parameters or buffers are not found in the checkpoint:
  [34mheads.bottleneck.0.{bias, weight, running_var, running_mean}[0m
  [34mheads.weight[0m
[05/08 09:14:29] fastreid.utils.checkpoint INFO: The checkpoint state_dict contains keys that are not used by the model:
  [35mpixel_mean[0m
  [35mpixel_std[0m
  [35mheads.bnneck.{weight, bias, running_mean, running_var, num_batches_tracked}[0m
  [35mheads.classifier.weight[0m
[05/08 09:14:29] fastreid.engine.defaults INFO: Prepare testing set
[05/08 09:14:30] fastreid.data.datasets.bases INFO: => Loaded MSMT17 in csv format: 
[36m| subset   | # ids   | # images   | # cameras   |
|:---------|:--------|:-----------|:------------|
| query    | 3060    | 11659      | 15          |
| gallery  | 3060    | 82161      | 15          |[0m
[05/08 09:14:30] fastreid.evaluation.evaluator INFO: Start inference on 93820 images
[05/08 09:14:57] fastreid.evaluation.evaluator INFO: Inference done 11/733. 0.7133 s / batch. ETA=0:08:36
[05/08 09:15:28] fastreid.evaluation.evaluator INFO: Inference done 53/733. 0.7251 s / batch. ETA=0:08:13
[05/08 09:15:58] fastreid.evaluation.evaluator INFO: Inference done 95/733. 0.7221 s / batch. ETA=0:07:41
[05/08 09:16:28] fastreid.evaluation.evaluator INFO: Inference done 137/733. 0.7215 s / batch. ETA=0:07:10
[05/08 09:16:33] fastreid INFO: Rank of current process: 0. World size: 1
[05/08 09:16:34] fastreid INFO: Environment info:
----------------------  ----------------------------------------------------------------------------------------------------
sys.platform            win32
Python                  3.9.16 (main, Mar  8 2023, 10:39:24) [MSC v.1916 64 bit (AMD64)]
numpy                   1.23.5
fastreid                1.3 @C:\source\GitRepo\reid\fast-reid\.\fastreid
FASTREID_ENV_MODULE     <not set>
PyTorch                 1.13.0 @C:\Users\wsyconan\.conda\envs\fastreid_2\lib\site-packages\torch
PyTorch debug build     False
GPU available           True
GPU 0                   NVIDIA GeForce GTX 1050 Ti
CUDA_HOME               C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7
Pillow                  9.4.0
torchvision             0.14.0+cpu @C:\Users\wsyconan\.conda\envs\fastreid_2\lib\site-packages\torchvision
torchvision arch flags  C:\Users\wsyconan\.conda\envs\fastreid_2\lib\site-packages\torchvision\_C.pyd; cannot find cuobjdump
cv2                     4.7.0
----------------------  ----------------------------------------------------------------------------------------------------
PyTorch built with:
  - C++ Version: 199711
  - MSVC 192829337
  - Intel(R) Math Kernel Library Version 2020.0.2 Product Build 20200624 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v2.6.0 (Git Hash 52b5f107dd9cf10910aaa19cb47f3abf9b349815)
  - OpenMP 2019
  - LAPACK is enabled (usually provided by MKL)
  - CPU capability usage: AVX2
  - CUDA Runtime 11.7
  - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_61,code=sm_61;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_37,code=compute_37
  - CuDNN 8.5
  - Magma 2.5.4
  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CUDA_VERSION=11.7, CUDNN_VERSION=8.5.0, CXX_COMPILER=C:/cb/pytorch_1000000000000/work/tmp_bin/sccache-cl.exe, CXX_FLAGS=/DWIN32 /D_WINDOWS /GR /EHsc /w /bigobj -DUSE_PTHREADPOOL -openmp:experimental -IC:/cb/pytorch_1000000000000/work/mkl/include -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DUSE_FBGEMM -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -DEDGE_PROFILER_USE_KINETO, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.13.0, USE_CUDA=ON, USE_CUDNN=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=OFF, USE_OPENMP=ON, USE_ROCM=OFF, 

[05/08 09:16:34] fastreid INFO: Command line arguments: Namespace(config_file='./configs/MSMT17/bagtricks_R50-ibn.yml', resume=False, eval_only=False, num_gpus=1, num_machines=1, machine_rank=0, dist_url='tcp://127.0.0.1:49153', opts=['MODEL.DEVICE', 'cuda:0'])
[05/08 09:16:34] fastreid INFO: Contents of args.config_file=./configs/MSMT17/bagtricks_R50-ibn.yml:
_BASE_: ../Base-bagtricks.yml

MODEL:
  BACKBONE:
    WITH_IBN: True

DATASETS:
  NAMES: ("MSMT17",)
  TESTS: ("MSMT17",)

OUTPUT_DIR: logs/msmt17/bagtricks_R50-ibn


[05/08 09:16:34] fastreid INFO: Running with full config:
CUDNN_BENCHMARK: True
DATALOADER:
  NUM_INSTANCE: 4
  NUM_WORKERS: 8
  SAMPLER_TRAIN: NaiveIdentitySampler
  SET_WEIGHT: []
DATASETS:
  COMBINEALL: False
  NAMES: ('MSMT17',)
  TESTS: ('MSMT17',)
INPUT:
  AFFINE:
    ENABLED: False
  AUGMIX:
    ENABLED: False
    PROB: 0.0
  AUTOAUG:
    ENABLED: False
    PROB: 0.0
  CJ:
    BRIGHTNESS: 0.15
    CONTRAST: 0.15
    ENABLED: False
    HUE: 0.1
    PROB: 0.5
    SATURATION: 0.1
  CROP:
    ENABLED: False
    RATIO: [0.75, 1.3333333333333333]
    SCALE: [0.16, 1]
    SIZE: [224, 224]
  FLIP:
    ENABLED: True
    PROB: 0.5
  PADDING:
    ENABLED: True
    MODE: constant
    SIZE: 10
  REA:
    ENABLED: True
    PROB: 0.5
    VALUE: [123.675, 116.28, 103.53]
  RPT:
    ENABLED: False
    PROB: 0.5
  SIZE_TEST: [256, 128]
  SIZE_TRAIN: [256, 128]
KD:
  EMA:
    ENABLED: False
    MOMENTUM: 0.999
  MODEL_CONFIG: []
  MODEL_WEIGHTS: []
MODEL:
  BACKBONE:
    ATT_DROP_RATE: 0.0
    DEPTH: 50x
    DROP_PATH_RATIO: 0.1
    DROP_RATIO: 0.0
    FEAT_DIM: 2048
    LAST_STRIDE: 1
    NAME: build_resnet_backbone
    NORM: BN
    PRETRAIN: True
    PRETRAIN_PATH: 
    SIE_COE: 3.0
    STRIDE_SIZE: (16, 16)
    WITH_IBN: True
    WITH_NL: False
    WITH_SE: False
  DEVICE: cuda:0
  FREEZE_LAYERS: []
  HEADS:
    CLS_LAYER: Linear
    EMBEDDING_DIM: 0
    MARGIN: 0.0
    NAME: EmbeddingHead
    NECK_FEAT: before
    NORM: BN
    NUM_CLASSES: 0
    POOL_LAYER: GlobalAvgPool
    SCALE: 1
    WITH_BNNECK: True
  LOSSES:
    CE:
      ALPHA: 0.2
      EPSILON: 0.1
      SCALE: 1.0
    CIRCLE:
      GAMMA: 128
      MARGIN: 0.25
      SCALE: 1.0
    COSFACE:
      GAMMA: 128
      MARGIN: 0.25
      SCALE: 1.0
    FL:
      ALPHA: 0.25
      GAMMA: 2
      SCALE: 1.0
    NAME: ('CrossEntropyLoss', 'TripletLoss')
    TRI:
      HARD_MINING: True
      MARGIN: 0.3
      NORM_FEAT: False
      SCALE: 1.0
  META_ARCHITECTURE: Baseline
  PIXEL_MEAN: [123.675, 116.28, 103.53]
  PIXEL_STD: [58.395, 57.120000000000005, 57.375]
  QUEUE_SIZE: 8192
  WEIGHTS: 
OUTPUT_DIR: logs/msmt17/bagtricks_R50-ibn
SOLVER:
  AMP:
    ENABLED: True
  BASE_LR: 0.00035
  BIAS_LR_FACTOR: 1.0
  CHECKPOINT_PERIOD: 30
  CLIP_GRADIENTS:
    CLIP_TYPE: norm
    CLIP_VALUE: 5.0
    ENABLED: False
    NORM_TYPE: 2.0
  DELAY_EPOCHS: 0
  ETA_MIN_LR: 1e-07
  FREEZE_ITERS: 0
  GAMMA: 0.1
  HEADS_LR_FACTOR: 1.0
  IMS_PER_BATCH: 64
  MAX_EPOCH: 120
  MOMENTUM: 0.9
  NESTEROV: False
  OPT: Adam
  SCHED: MultiStepLR
  STEPS: [40, 90]
  WARMUP_FACTOR: 0.1
  WARMUP_ITERS: 2000
  WARMUP_METHOD: linear
  WEIGHT_DECAY: 0.0005
  WEIGHT_DECAY_BIAS: 0.0005
  WEIGHT_DECAY_NORM: 0.0005
TEST:
  AQE:
    ALPHA: 3.0
    ENABLED: False
    QE_K: 5
    QE_TIME: 1
  EVAL_PERIOD: 30
  FLIP:
    ENABLED: False
  IMS_PER_BATCH: 128
  METRIC: cosine
  PRECISE_BN:
    DATASET: Market1501
    ENABLED: False
    NUM_ITER: 300
  RERANK:
    ENABLED: False
    K1: 20
    K2: 6
    LAMBDA: 0.3
  ROC:
    ENABLED: False
[05/08 09:16:34] fastreid INFO: Full config saved to C:\source\GitRepo\reid\fast-reid\logs\msmt17\bagtricks_R50-ibn\config.yaml
[05/08 09:16:34] fastreid.utils.env INFO: Using a generated random seed 34341372
[05/08 09:16:34] fastreid.engine.defaults INFO: Prepare training set
[05/08 09:16:35] fastreid.data.datasets.bases INFO: => Loaded MSMT17 in csv format: 
| subset   | # ids   | # images   | # cameras   |
|:---------|:--------|:-----------|:------------|
| train    | 1041    | 30248      | 15          |
[05/08 09:16:35] fastreid.data.build INFO: Using training sampler NaiveIdentitySampler
[05/08 09:16:35] fastreid.engine.defaults INFO: Auto-scaling the num_classes=1041
[05/08 09:16:35] fastreid.modeling.backbones.resnet INFO: Loading pretrained model from C:\Users\wsyconan/.cache\torch\checkpoints\resnet50_ibn_a-d9d0bb7b.pth
[05/08 09:16:36] fastreid.modeling.backbones.resnet INFO: The checkpoint state_dict contains keys that are not used by the model:
  fc.{weight, bias}
[05/08 09:16:39] fastreid.engine.defaults INFO: Model:
Baseline(
  (backbone): ResNet(
    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (bn1): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
    (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
    (layer1): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
        (downsample): Sequential(
          (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (2): Bottleneck(
        (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
    )
    (layer2): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn2): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
        (downsample): Sequential(
          (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (2): Bottleneck(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (3): Bottleneck(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
    )
    (layer3): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn2): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
        (downsample): Sequential(
          (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (2): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (3): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (4): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (5): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): IBN(
          (IN): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (BN): BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
    )
    (layer4): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
        (downsample): Sequential(
          (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
      (2): Bottleneck(
        (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (se): Identity()
      )
    )
  )
  (heads): EmbeddingHead(
    (pool_layer): GlobalAvgPool(output_size=1)
    (bottleneck): Sequential(
      (0): BatchNorm(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (cls_layer): Linear(num_classes=1041, scale=1, margin=0.0)
  )
)
