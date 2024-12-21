# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
 ICCV2021
 https://arxiv.org/abs/2103.14030

## Abstract
- 다양한 Vision Task에 적합한 구조.
- Local Window를 적용하여 inductive bias 개선.
- Patch Merging을 통해 레이어 간 계층적 구조를 형성하여 이미지 특성 고려.


## Introduction
![teaser](figures/teaser.png)
비전 분야에서는 주로 CNN 백본을 사용해왔으나 NLP분야에서 Transformer가 높은 성능을 보인 이후 비전 분야에도 Transformer를 적용한 ViT모델이 나왔습니다. Self-Attention기법을 사용하였기에 이미지의 전역적인 정보를 가져오는 것이 가능하여 classification작업은 적용이 가능했으나 해상도가 높은 이미지에 대해서 quadratic한 연산량을 보여 효율성이 떨어졌고 detection이나 segmentation과 같은 정교한 비전Task에 대해서는 적용이 불가했습니다.   

따라서 이를 해결하기 위해 제안된 Swin transformer는 두가지 특징을 가집니다.

**1. Hierarchical Feature maps**

Swin Transformer는 레이어가 깊어질수록 회색 테두리로 되어있는 이미지 패치들을 병합하고 빨간색 테두리로 되어있는 윈도우 안에서 지역적인 Self-Attention 연산을 하여 입력 이미지에 대해 선형적인 연산 복잡도를 갖게합니다. 각 윈도우의 패치 수는 고정되어 있으므로 계산 복잡도는 이미지 크기에 비례합니다.
![](https://velog.velcdn.com/images/bh9711/post/71b8da00-8907-4a25-81bc-d3aa1e42ca29/image.png)
이러한 계층적 특징 맵을 통해 Swin Transformer는 다양한 해상도에서 이미지 정보를 가져올 수 있어 정교한 비전 Task도 수행 가능하게 합니다. ResNet과 같이 FPN(Feature Pyramid Network)구조이기 때문에 패치를 증가 시켜도 선형적으로 연산량이 증가하게 됩니다. 

**2. Shifted Window self-attention**







계산 복잡도를 선형적으로 줄이기 위해 Swin transformer는 비중첩 윈도우 내에서 self-attention 연산을 하고 윈도우 경계를 넘는 정보를 교환하기 위해 윈도우를 이동시킨다.

**self-attention연산**은 트랜스포머의 인코더와 디코더 블록 모두에서 수행되며 쿼리, 키, 밸류 3개 요소 사이의 문맥적 관계성을 추출하는 과정이다.

먼저 쿼리, 키, 밸류를 만든다. 입력 벡터 수열에 쿼리, 키, 벨류를 만들어주는 행렬곱하여 각각의 행렬을 만들고 이는 태스크를 가장 잘 수행하는 방향으로 학습이 업데이트된다고 한다.

다음으로 쿼리의 셀프 어텐션 출력값 계산은 쿼리와 키를 행렬곱한 뒤 해당 행렬의 모든 요소값을 키 차원수의 제곱근 값으로 나눠주고 이 행렬을 행 단위로 소프트맥스를 취해 스코어 행렬을 만들어준다. 

그 다음 스코어 행렬에 V를 행렬곱해줘서 셀프 어텐션 계산을 마친다.
![](https://velog.velcdn.com/images/bh9711/post/ff6f141f-68ca-4dad-9d3a-735808f1e459/image.png)

Swin Transformer의 핵심 설계 요소는 아래 그림과 같이 연속적인 self-attention Layer 사이의 window 파티션의 이동이다. Shifted window는 이전 Layer의 window를 연결하여 모델링 능력을 크게 향상시키는 연결을 제공한다. 이 전략은 또한 실제 지연 시간과 관련하여 효율적이다. Window 내의 모든 쿼리 패치는 하드웨어에서 메모리 액세스를 용이하게 하는 동일한 키 세트를 공유한다.


![](https://velog.velcdn.com/images/bh9711/post/790d1bc5-936f-4596-a1af-b1f9763e04aa/image.png)


이전의 슬라이딩 window 기반 self-attention 접근 방식은 다른 쿼리 픽셀에 대한 다른 키 세트로 인해 일반 하드웨어에서 낮은 지연 시간으로 어려움을 겪고 있다. 제안된 shifted window 방식이 sliding window 방식보다 지연 시간이 훨씬 짧지만 모델링 능력은 비슷하다. 또한 shfted window 접근 방식은 모든 MLP 아키텍처에도 유익하다.

## Method

### 1. Swin transformer 구조

![](https://velog.velcdn.com/images/bh9711/post/1ecdfa56-9b64-496b-b34f-1449694e2e4f/image.png)

위 그림은 Swin Transformer의 모델 구조와 transformer block을 나타낸다. ViT와 같은 patch partition에 의해 입력 RGB 이미지를 겹치지 않는 패치로 분할한다.

**patch partition**에서는 가장 작은 grid 단위로 image를 partition 한다. 이 패치들은 토큰으로 간주된다. 본 연구에서는 4x4 크기의 patch를 사용하여 각 패치의 feature 차원은 4x4x3=48 이라고 한다.

선형 임베딩 layer에서는 feature를 임의의 크기의 차원 C로 정사영 시킨다. 이러한 패치 토큰에는 수정된 self-attention 연산이 포함된 여러 Transformer 블록이 이러한 패치 토큰에 적용된다. 이는 선형 임베딩과 함께 1단계라고 한다.

계층적인 구조를 위해 네트워크의 계층이 깊어질수록 patch merging layer에 의해 토큰의 수가 감소한다. 

첫번째 patch merging layer는 2개x2개의 neighboring patch로 구성된 각 그룹의 특징을 concat하여 4C-dimentional feature에 선형layer를 적용한다.

feature 변환은 patch merging과 함께 2단계라고 하며 feature변환 이후 Swin transformer block이 적용된다. 

Swin Transformer block에서는 두 개의 연속된 Swin Transformer block이 적용되며, 각 patch에 대한 attention을 연산한다.

첫 번째 transformer block에서는 window based self-attention 이 계산되고, 두번째 block에서 shifted window based self-attention이 적용된다. 

이 process는 3단계와 4단계에서도 반복되는 구조이다.

### 2. Shifted Window based Self-Attention

표준 Transformer 아키텍처와 image classification을 위한 버전은 토큰과 다른 모든 토큰 간의 관계가 계산되는 글로벌 self-attention을 수행한다. 글로벌 계산은 토큰 수와 관련하여 2차 복잡도를 초래하여 조밀한 예측이나 고해상도 이미지를 나타내기 위해 막대한 토큰 세트가 필요한 많은 비전 문제에 적합하지 않다.

#### 1. Self-attention in non-overlapped window
효율적인 모델링을 위해 전체영역이 아닌 local window내에서 self-attention을 계산하였다. 
![](https://velog.velcdn.com/images/bh9711/post/c35a7584-2b73-45ad-8215-1c687405fcd8/image.png)

기존의 self-attention과 window based self-attention의 계산식이며 window내에는 MxM개의 패치가 존재하는데 MSA는 패치 수 인 hw에 대해 2차이고 W-MSA는 M이 고정된 경우 hw에 대해 선형이다. 


#### 2. Shifted window partitioning in successive blocks
window based self-attention은 window간의 연결이 부족하여 모델링 능력이 제한된다고 한다. 따라서 겹치지 않는 효율적인 계산을 유지하면서 window사이의 연결을 위해 연속되는 Swin transformer블록에서 두 개의 파티션 구성을 번갈아 가며 전환하는 **Shifted window partitioning** 방식을 제안했다.

Shifted window partitioning 접근법은 이전 레이어에서 인접한 겹치지 않는 window 사이의 연결을 도입하고 image classification, object detection, semantic segmentation에 효과적이라고 한다.


#### 3. Efficient batch computation for shifted configuration
shifted window partitioning의 문제는 더 많은 window를 만들며 일부 창은 M x M보다 작아야 하기 때문에 효율적인 계산을 위해 본 논문에서는 **cyclic-shift**를 적용하였다고 한다. 

cyclic shift에 대해 간략히 설명하자면 아래 그림과 같다.
![image](https://github.com/user-attachments/assets/c65c4b14-0fdc-4812-ac5e-a7407f08f516)

![](https://velog.velcdn.com/images/bh9711/post/fa770d40-b8b5-4488-8dcc-7b574fb0ce49/image.png)

위의 그림과 같이 왼쪽 위 방향으로 cyclic-shifting하여 보다 효율적인 배치 계산 방식을 제안하여 배치된 window의 수가 일반 window partition과 동일하게 유지되어 효율적이라고 한다.

색깔부분과 회색부분은 서로 다르게 attention이 적용되어야 하기 때문에 mask를 적용을 해준다.

이 attention mask는 서로 인접하지 않은 패치들끼리 연산을 하게 만들고 인접하지 않은 패치들에 대해 가중치를 0을 만들어 해당부분의 정보를 무시하는 기술이다.

따라서 인접하지 않은 패치들 사이의 상호작용을 제한하고 관련있는 패치들끼리의 attention에 집중할 수 있다.


#### 4. Relative position bias
Swin Transformer에서는 패치들 간의 상대적인 위치 정보를 수집하여 저장한다. 이 정보를 활용하면 패치 간 거리에 따라 가중치를 부여하여 자연어 처리에서 사용되는 어텐션 메커니즘과 유사하게, 
이미지 내에서 더 먼 패치들과의 상호작용을 조절할 수 있다. 정리하자면 **relative position bias는 패치 간 상대적인 위치에 대한 편향 정보를 나타내는 개념으로 self-attention매커니즘에 적용되어 패치 간의 상대적인 위치에 따른 중요도를 반영한다.**

swin은 vit와 달리 입력시퀀스에 position embedding을 더해주지 않고 self-attention을 수행하는 과정에서 relative position bias를 더해준다.
![](https://velog.velcdn.com/images/bh9711/post/e8d31827-166c-4e71-b66c-1041f584a278/image.png)![](https://velog.velcdn.com/images/bh9711/post/ccd34f16-65a4-4c2e-a5b7-38ca25e5e735/image.png)

두 축마다 relative Position의 범위는 [-M + 1, M - 1]이다.
윈도우 크기가 3인 matrix의 범위는 [-2 , 2]가 된다.
![](https://velog.velcdn.com/images/bh9711/post/9dd31320-3075-4ee9-b0f8-d4cfdb0963a7/image.png)

x축과 y축을 기점으로 패치 간 상대적인 위치에 대한 편향정보를 계산한다. 이 과정을 거쳐서 최종 relative position bias를 얻게 된다.

![](https://velog.velcdn.com/images/bh9711/post/3f9cb402-2ba0-4920-9b66-6960404544aa/image.png)

bias matrix의 크기가 25이기 때문에 범위가 0번 인덱스부터 24번까지 있는 것을 확인할 수 있으며 대각행렬이 모두 12인 것을 확인할 수 있다.
그러나 패치 간의 상대적인 위치에 대한 차원이 2M-1로 적기 때문에 정확한 편향정보를 끌어내기 어렵다. 
따라서 본 논문에서는 보다 적은 학습 파라미터로 넓은 범위의 relative position bias를 제안하였다.

이를 통해 swin은 긴 범위의 이미지를 효과적으로 학습하고 연산 복잡도를 크게 늘리지 않고 성능을 개선할 수 있다고 한다.
 
![image](https://github.com/user-attachments/assets/b0dfb775-512d-4ad2-bb92-086b3618c29e)

이렇게 구성된 x축과 y축의 매트릭스에 각각 윈도우사이즈에 -1한 값을 모든 element에 적용하여 더해주게 된다. 이렇게 하는 이유는 이 값들을 실제 인덱스로 표기하기 위해서 값의 범위가 0부터 시작하게 만드는 것이다.
그 다음으로 x축의 매트릭스에 모든 element의 윈도우 사이즈를 2배 한 결과에 -1이란 값을 적용을 해주고 모두 곱해준다. 그리고 이 결과에 y축에 대한 매트릭스를 더해주게 된다. 이렇게 하면 아래와 같은 relative position bias가 나나타게 된다.

![image](https://github.com/user-attachments/assets/dc5c9cc8-15a9-44e4-858c-994c12c79321)



## 실험결과
Swin Transformer는 COCO object detection(`58.7 box AP` and `51.1 mask AP` on test-dev)와 ADE20K semantic segmentation (`53.5 mIoU` on val)에서 강력한 성능을 달성하여 이전 모델보다 큰 성능 증가를 보였습니다.
**ImageNet-1K and ImageNet-22K Pretrained Swin-V1 Models**

| name | pretrain | resolution |acc@1 | acc@5 | #params | FLOPs | FPS| 22K model | 1K model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |:---: |
| Swin-T | ImageNet-1K | 224x224 | 81.2 | 95.5 | 28M | 4.5G | 755 | - | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth)/[baidu](https://pan.baidu.com/s/156nWJy4Q28rDlrX-rRbI3w)/[config](configs/swin/swin_tiny_patch4_window7_224.yaml)/[log](https://github.com/SwinTransformer/storage/files/7745562/log_swin_tiny_patch4_window7_224.txt) |
| Swin-S | ImageNet-1K | 224x224 | 83.2 | 96.2 | 50M | 8.7G | 437 | - | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth)/[baidu](https://pan.baidu.com/s/1KFjpj3Efey3LmtE1QqPeQg)/[config](configs/swin/swin_small_patch4_window7_224.yaml)/[log](https://github.com/SwinTransformer/storage/files/7745563/log_swin_small_patch4_window7_224.txt) |
| Swin-B | ImageNet-1K | 224x224 | 83.5 | 96.5 | 88M | 15.4G | 278  | - | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth)/[baidu](https://pan.baidu.com/s/16bqCTEc70nC_isSsgBSaqQ)/[config](configs/swin/swin_base_patch4_window7_224.yaml)/[log](https://github.com/SwinTransformer/storage/files/7745564/log_swin_base_patch4_window7_224.txt) |
| Swin-B | ImageNet-1K | 384x384 | 84.5 | 97.0 | 88M | 47.1G | 85 | - | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth)/[baidu](https://pan.baidu.com/s/1xT1cu740-ejW7htUdVLnmw)/[config](configs/swin/swin_base_patch4_window12_384_finetune.yaml) |
| Swin-T | ImageNet-22K | 224x224 | 80.9 | 96.0 | 28M | 4.5G | 755 | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth)/[baidu](https://pan.baidu.com/s/1vct0VYwwQQ8PYkBjwSSBZQ?pwd=swin)/[config](configs/swin/swin_tiny_patch4_window7_224_22k.yaml) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22kto1k_finetune.pth)/[baidu](https://pan.baidu.com/s/1K0OO-nGZDPkR8fm_r83e8Q?pwd=swin)/[config](configs/swin/swin_tiny_patch4_window7_224_22kto1k_finetune.yaml) |
| Swin-S | ImageNet-22K | 224x224 | 83.2 | 97.0 | 50M | 8.7G | 437 | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22k.pth)/[baidu](https://pan.baidu.com/s/11NC1xdT5BAGBgazdTme5Sg?pwd=swin)/[config](configs/swin/swin_small_patch4_window7_224_22k.yaml) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22kto1k_finetune.pth)/[baidu](https://pan.baidu.com/s/10RFVfjQJhwPfeHrmxQUaLw?pwd=swin)/[config](configs/swin/swin_small_patch4_window7_224_22kto1k_finetune.yaml) |
| Swin-B | ImageNet-22K | 224x224 | 85.2 | 97.5 | 88M | 15.4G | 278 | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth)/[baidu](https://pan.baidu.com/s/1y1Ec3UlrKSI8IMtEs-oBXA)/[config](configs/swin/swin_base_patch4_window7_224_22k.yaml) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth)/[baidu](https://pan.baidu.com/s/1n_wNkcbRxVXit8r_KrfAVg)/[config](configs/swin/swin_base_patch4_window7_224_22kto1k_finetune.yaml) |
| Swin-B | ImageNet-22K | 384x384 | 86.4 | 98.0 | 88M | 47.1G | 85 | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth)/[baidu](https://pan.baidu.com/s/1vwJxnJcVqcLZAw9HaqiR6g) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22kto1k.pth)/[baidu](https://pan.baidu.com/s/1caKTSdoLJYoi4WBcnmWuWg)/[config](configs/swin/swin_base_patch4_window12_384_22kto1k_finetune.yaml) |
| Swin-L | ImageNet-22K | 224x224 | 86.3 | 97.9 | 197M | 34.5G | 141 | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth)/[baidu](https://pan.baidu.com/s/1pws3rOTFuOebBYP3h6Kx8w)/[config](configs/swin/swin_large_patch4_window7_224_22k.yaml) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22kto1k.pth)/[baidu](https://pan.baidu.com/s/1NkQApMWUhxBGjk1ne6VqBQ)/[config](configs/swin/swin_large_patch4_window7_224_22kto1k_finetune.yaml) |
| Swin-L | ImageNet-22K | 384x384 | 87.3 | 98.2 | 197M | 103.9G | 42 | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth)/[baidu](https://pan.baidu.com/s/1sl7o_bJA143OD7UqSLAMoA) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22kto1k.pth)/[baidu](https://pan.baidu.com/s/1X0FLHQyPOC6Kmv2CmgxJvA)/[config](configs/swin/swin_large_patch4_window12_384_22kto1k_finetune.yaml) |

**ImageNet-1K and ImageNet-22K Pretrained Swin-V2 Models**

| name | pretrain | resolution | window |acc@1 | acc@5 | #params | FLOPs | FPS |22K model | 1K model |
|:---------------------:| :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---:|:---: |:---: |
| SwinV2-T | ImageNet-1K | 256x256 | 8x8 | 81.8 | 95.9 | 28M | 5.9G | 572 | - | [github](https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_tiny_patch4_window8_256.pth)/[baidu](https://pan.baidu.com/s/1RzLkAH_5OtfRCJe6Vlg6rg?pwd=swin)/[config](configs/swinv2/swinv2_tiny_patch4_window8_256.yaml) |
| SwinV2-S | ImageNet-1K | 256x256 | 8x8 | 83.7 | 96.6 | 50M | 11.5G | 327 | - | [github](https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_small_patch4_window8_256.pth)/[baidu](https://pan.baidu.com/s/195PdA41szEduW3jEtRSa4Q?pwd=swin)/[config](configs/swinv2/swinv2_small_patch4_window8_256.yaml) |
| SwinV2-B | ImageNet-1K | 256x256 | 8x8 | 84.2 | 96.9 | 88M | 20.3G | 217 | - | [github](https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window8_256.pth)/[baidu](https://pan.baidu.com/s/18AfMSz3dPyzIvP1dKuERvQ?pwd=swin)/[config](configs/swinv2/swinv2_base_patch4_window8_256.yaml) |
| SwinV2-T | ImageNet-1K | 256x256 | 16x16 | 82.8 | 96.2 | 28M | 6.6G | 437 | - | [github](https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_tiny_patch4_window16_256.pth)/[baidu](https://pan.baidu.com/s/1dyK3cK9Xipmv6RnTtrPocw?pwd=swin)/[config](configs/swinv2/swinv2_tiny_patch4_window16_256.yaml) |
| SwinV2-S | ImageNet-1K | 256x256 | 16x16 | 84.1 | 96.8 | 50M | 12.6G  | 257 | - | [github](https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_small_patch4_window16_256.pth)/[baidu](https://pan.baidu.com/s/1ZIPiSfWNKTPp821Ka-Mifw?pwd=swin)/[config](configs/swinv2/swinv2_small_patch4_window16_256.yaml) |
| SwinV2-B | ImageNet-1K | 256x256 | 16x16 | 84.6 | 97.0 | 88M | 21.8G | 174 | - | [github](https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window16_256.pth)/[baidu](https://pan.baidu.com/s/1dlDQGn8BXCmnh7wQSM5Nhw?pwd=swin)/[config](configs/swinv2/swinv2_base_patch4_window16_256.yaml) |
| SwinV2-B<sup>\*</sup> | ImageNet-22K | 256x256 | 16x16 | 86.2 | 97.9 |  88M | 21.8G | 174 | [github](https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window12_192_22k.pth)/[baidu](https://pan.baidu.com/s/1Xc2rsSsRQz_sy5mjgfxrMQ?pwd=swin)/[config](configs/swinv2/swinv2_base_patch4_window12_192_22k.yaml) | [github](https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth)/[baidu](https://pan.baidu.com/s/1sgstld4MgGsZxhUAW7MlmQ?pwd=swin)/[config](configs/swinv2/swinv2_base_patch4_window12to16_192to256_22kto1k_ft.yaml) |
| SwinV2-B<sup>\*</sup> | ImageNet-22K | 384x384 | 24x24 | 87.1 | 98.2 | 88M | 54.7G | 57  | [github](https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window12_192_22k.pth)/[baidu](https://pan.baidu.com/s/1Xc2rsSsRQz_sy5mjgfxrMQ?pwd=swin)/[config](configs/swinv2/swinv2_base_patch4_window12_192_22k.yaml) | [github](https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth)/[baidu](https://pan.baidu.com/s/17u3sEQaUYlvfL195rrORzQ?pwd=swin)/[config](configs/swinv2/swinv2_base_patch4_window12to24_192to384_22kto1k_ft.yaml) |
| SwinV2-L<sup>\*</sup> | ImageNet-22K | 256x256 | 16x16 | 86.9 | 98.0 | 197M | 47.5G | 95  | [github](https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_large_patch4_window12_192_22k.pth)/[baidu](https://pan.baidu.com/s/11PhCV7qAGXtZ8dXNgyiGOw?pwd=swin)/[config](configs/swinv2/swinv2_large_patch4_window12_192_22k.yaml) | [github](https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_large_patch4_window12to16_192to256_22kto1k_ft.pth)/[baidu](https://pan.baidu.com/s/1pqp31N80qIWjFPbudzB6Bw?pwd=swin)/[config](configs/swinv2/swinv2_large_patch4_window12to16_192to256_22kto1k_ft.yaml) |
| SwinV2-L<sup>\*</sup> | ImageNet-22K | 384x384 | 24x24 | 87.6 | 98.3 | 197M | 115.4G | 33  | [github](https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_large_patch4_window12_192_22k.pth)/[baidu](https://pan.baidu.com/s/11PhCV7qAGXtZ8dXNgyiGOw?pwd=swin)/[config](configs/swinv2/swinv2_large_patch4_window12_192_22k.yaml) | [github](https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_large_patch4_window12to24_192to384_22kto1k_ft.pth)/[baidu](https://pan.baidu.com/s/13URdNkygr3Xn0N3e6IwjgA?pwd=swin)/[config](configs/swinv2/swinv2_large_patch4_window12to24_192to384_22kto1k_ft.yaml) |

Note: 
- SwinV2-B<sup>\*</sup>  (SwinV2-L<sup>\*</sup>) with input resolution of 256x256 and 384x384 both fine-tuned from the same pre-training model using a smaller input resolution of 192x192.
- SwinV2-B<sup>\*</sup> (384x384) achieves 78.08 acc@1 on ImageNet-1K-V2 while SwinV2-L<sup>\*</sup> (384x384) achieves 78.31.

**ImageNet-1K Pretrained Swin MLP Models**

| name | pretrain | resolution |acc@1 | acc@5 | #params | FLOPs | FPS |  1K model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [Mixer-B/16](https://arxiv.org/pdf/2105.01601.pdf) | ImageNet-1K | 224x224 | 76.4 | - | 59M | 12.7G | - | [official repo](https://github.com/google-research/vision_transformer) |
| [ResMLP-S24](https://arxiv.org/abs/2105.03404) | ImageNet-1K | 224x224 | 79.4 | - | 30M | 6.0G | 715 | [timm](https://github.com/rwightman/pytorch-image-models) |
| [ResMLP-B24](https://arxiv.org/abs/2105.03404) | ImageNet-1K | 224x224 | 81.0 | - | 116M | 23.0G |  231 | [timm](https://github.com/rwightman/pytorch-image-models) |
| Swin-T/C24 | ImageNet-1K | 256x256 | 81.6 | 95.7 | 28M | 5.9G | 563 | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.5/swin_tiny_c24_patch4_window8_256.pth)/[baidu](https://pan.baidu.com/s/17k-7l6Sxt7uZ7IV0f26GNQ)/[config](configs/swin/swin_tiny_c24_patch4_window8_256.yaml) |
| SwinMLP-T/C24 | ImageNet-1K | 256x256 | 79.4 | 94.6 | 20M | 4.0G | 807 | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.5/swin_mlp_tiny_c24_patch4_window8_256.pth)/[baidu](https://pan.baidu.com/s/1Sa4vP5R0M2RjfIe9HIga-Q)/[config](configs/swin/swin_mlp_tiny_c24_patch4_window8_256.yaml) |
| SwinMLP-T/C12 | ImageNet-1K | 256x256 | 79.6 | 94.7 | 21M | 4.0G | 792 | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.5/swin_mlp_tiny_c12_patch4_window8_256.pth)/[baidu](https://pan.baidu.com/s/1mM9J2_DEVZHUB5ASIpFl0w)/[config](configs/swin/swin_mlp_tiny_c12_patch4_window8_256.yaml) |
| SwinMLP-T/C6 | ImageNet-1K | 256x256 | 79.7 | 94.9 | 23M | 4.0G | 766 | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.5/swin_mlp_tiny_c6_patch4_window8_256.pth)/[baidu](https://pan.baidu.com/s/1hUTYVT2W1CsjICw-3W-Vjg)/[config](configs/swin/swin_mlp_tiny_c6_patch4_window8_256.yaml) |
| SwinMLP-B | ImageNet-1K | 224x224 | 81.3 | 95.3 | 61M | 10.4G | 409 | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.5/swin_mlp_base_patch4_window7_224.pth)/[baidu](https://pan.baidu.com/s/1zww3dnbX3GxNiGfb-GwyUg)/[config](configs/swin/swin_mlp_base_patch4_window7_224.yaml) |

Note: access code for `baidu` is `swin`. C24 means each head has 24 channels.

**ImageNet-22K Pretrained Swin-MoE Models**

- Please refer to [get_started](get_started.md#mixture-of-experts-support) for instructions on running Swin-MoE. 
- Pretrained models for Swin-MoE can be found in [MODEL HUB](MODELHUB.md#imagenet-22k-pretrained-swin-moe-models)

## Main Results on Downstream Tasks

**COCO Object Detection (2017 val)**

| Backbone | Method | pretrain | Lr Schd | box mAP | mask mAP | #params | FLOPs |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Swin-T | Mask R-CNN | ImageNet-1K | 3x | 46.0 | 41.6 | 48M | 267G |
| Swin-S | Mask R-CNN | ImageNet-1K | 3x | 48.5 | 43.3 | 69M | 359G |
| Swin-T | Cascade Mask R-CNN | ImageNet-1K | 3x | 50.4 | 43.7 | 86M | 745G |
| Swin-S | Cascade Mask R-CNN | ImageNet-1K |  3x | 51.9 | 45.0 | 107M | 838G |
| Swin-B | Cascade Mask R-CNN | ImageNet-1K |  3x | 51.9 | 45.0 | 145M | 982G |
| Swin-T | RepPoints V2 | ImageNet-1K | 3x | 50.0 | - | 45M | 283G |
| Swin-T | Mask RepPoints V2 | ImageNet-1K | 3x | 50.3 | 43.6 | 47M | 292G |
| Swin-B | HTC++ | ImageNet-22K | 6x | 56.4 | 49.1 | 160M | 1043G |
| Swin-L | HTC++ | ImageNet-22K | 3x | 57.1 | 49.5 | 284M | 1470G |
| Swin-L | HTC++<sup>*</sup> | ImageNet-22K | 3x | 58.0 | 50.4 | 284M | - |

Note: <sup>*</sup> indicates multi-scale testing.

**ADE20K Semantic Segmentation (val)**

| Backbone | Method | pretrain | Crop Size | Lr Schd | mIoU | mIoU (ms+flip) | #params | FLOPs |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Swin-T | UPerNet | ImageNet-1K | 512x512 | 160K | 44.51 | 45.81 | 60M | 945G |
| Swin-S | UperNet | ImageNet-1K | 512x512 | 160K | 47.64 | 49.47 | 81M | 1038G |
| Swin-B | UperNet | ImageNet-1K | 512x512 | 160K | 48.13 | 49.72 | 121M | 1188G |
| Swin-B | UPerNet | ImageNet-22K | 640x640 | 160K | 50.04 | 51.66 | 121M | 1841G |
| Swin-L | UperNet | ImageNet-22K | 640x640 | 160K | 52.05 | 53.53 | 234M | 3230G |


## Getting Started

- For **Image Classification**, please see [get_started.md](get_started.md) for detailed instructions.
- For **Object Detection and Instance Segmentation**, please see [Swin Transformer for Object Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection).
- For **Semantic Segmentation**, please see [Swin Transformer for Semantic Segmentation](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation).
- For **Self-Supervised Learning**, please see [Transformer-SSL](https://github.com/SwinTransformer/Transformer-SSL).
- For **Video Recognition**, please see [Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer).

- ## 참고자료
- https://www.youtube.com/watch?v=2lZvuU_IIMA&t=1813s
- https://www.youtube.com/watch?v=dFwmjV7wIKY&t=2s
- https://hchoi256.github.io/aipapercv/swin-transformer/

