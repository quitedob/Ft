
## News
- 2024.12.8 完善多并发，显存不随并发数增加
- 2024.12.21 添加wav2lip、musetalk模型预热，解决第一次推理卡顿问题。感谢[@heimaojinzhangyz](https://github.com/heimaojinzhangyz)
- 2024.12.28 添加数字人模型Ultralight-Digital-Human。 感谢[@lijihua2017](https://github.com/lijihua2017)
- 2025.2.7 添加fish-speech tts
- 2025.2.21 添加wav2lip256开源模型 感谢@不蠢不蠢
- 2025.3.2 添加腾讯语音合成服务
- 2025.3.16 支持mac gpu推理，感谢[@GcsSloop](https://github.com/GcsSloop) 
- 2025.5.1 精简运行参数，ernerf模型移至git分支ernerf-rtmp
- 2025.6.7 添加虚拟摄像头输出
- 2025.7.5 添加豆包语音合成, 感谢[@ELK-milu](https://github.com/ELK-milu)

## Features
1. 支持多种数字人模型: ernerf、musetalk、wav2lip、Ultralight-Digital-Human
2. 支持声音克隆
3. 支持数字人说话被打断
4. 支持全身视频拼接
5. 支持webrtc、虚拟摄像头输出
6. 支持动作编排：不说话时播放自定义视频
7. 支持多并发

## 1. Installation

Tested on Ubuntu 24.04, Python3.10, Pytorch 2.5.0 and CUDA 12.4

### 1.1 Install dependency

```bash
conda create -n nerfstream python=3.10
conda activate nerfstream
#如果cuda版本不为12.4(运行nvidia-smi确认版本)，根据<https://pytorch.org/get-started/previous-versions/>安装对应版本的pytorch 
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
#如果需要训练ernerf模型，安装下面的库
# pip install "git+https://github.com/facebookresearch/pytorch3d.git"
# pip install tensorflow-gpu==2.8.0
# pip install --upgrade "protobuf<=3.20.1"
``` 
安装常见问题[FAQ](https://livetalking-doc.readthedocs.io/zh-cn/latest/faq.html)  
linux cuda环境搭建可以参考这篇文章 <https://zhuanlan.zhihu.com/p/674972886>  
视频连不上解决方法 <https://mp.weixin.qq.com/s/MVUkxxhV2cgMMHalphr2cg>


## 2. Quick Start
- 下载模型
将wav2lip256.pth拷到本项目的models下, 重命名为wav2lip.pth;  
将wav2lip256_avatar1.tar.gz解压后整个文件夹拷到本项目的data/avatars下
- 运行  
python app.py --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1  
<font color=red>服务端需要开放端口 tcp:8010; udp:1-65536 </font>  
客户端可以选用以下两种方式:  
(1)用浏览器打开http://serverip:8010/webrtcapi.html , 先点‘start',播放数字人视频；然后在文本框输入任意文字，提交。数字人播报该段文字  

如果访问不了huggingface，在运行前
```
export HF_ENDPOINT=https://hf-mirror.com
``` 


## 3. More Usage
使用说明: <https://livetalking-doc.readthedocs.io/>
  
## 4. Docker Run  
不需要前面的安装，直接运行。
```
docker run --gpus all -it --network=host --rm registry.cn-beijing.aliyuncs.com/codewithgpu2/lipku-metahuman-stream:2K9qaMBu8v
```
代码在/root/metahuman-stream，先git pull拉一下最新代码，然后执行命令同第2、3步 

提供如下镜像
- autodl镜像: <https://www.codewithgpu.com/i/lipku/metahuman-stream/base>   
[autodl教程](https://livetalking-doc.readthedocs.io/en/latest/autodl/README.html)
- ucloud镜像: <https://www.compshare.cn/images/4458094e-a43d-45fe-9b57-de79253befe4?referral_code=3XW3852OBmnD089hMMrtuU&ytag=GPU_GitHub_livetalking>  
可以开放任意端口，不需要另外部署srs服务.  
[ucloud教程](https://livetalking-doc.readthedocs.io/en/latest/ucloud/ucloud.html) 


## 5. 性能
- 性能主要跟cpu和gpu相关，每路视频压缩需要消耗cpu，cpu性能与视频分辨率正相关；每路口型推理跟gpu性能相关。  
- 不说话时的并发数跟cpu相关，同时说话的并发数跟gpu相关。  
- 后端日志inferfps表示显卡推理帧率，finalfps表示最终推流帧率。两者都要在25以上才能实时。如果inferfps在25以上，finalfps达不到25表示cpu性能不足。  
- 实时推理性能
wav2lip256显卡3060以上即可，musetalk需要3080Ti以上。 

