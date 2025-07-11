# whisper_asr.py
# 基于Faster-Whisper的ASR实现，严格遵循BaseASR接口

from baseasr import BaseASR
import numpy as np
from faster_whisper import WhisperModel
import torch
from logger import logger # 使用项目统一的logger

class WhisperASR(BaseASR):
    """
    实现 BaseASR 接口，使用 Faster-Whisper
    """
    def __init__(self, opt):
        """
        初始化模型
        """
        super().__init__(opt) # 调用父类的构造函数
        self.opt = opt

        model_size = getattr(self.opt, 'whisper_model_size', 'base')

        # 根据CUDA可用性选择设备和计算类型
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16" # 或者 "int8_float16"
            logger.info("检测到 CUDA，WhisperASR 将使用 GPU。")
        else:
            device = "cpu"
            compute_type = "int8" # 或者 "float32"
            logger.info("未检测到 CUDA，WhisperASR 将使用 CPU。")

        logger.info(f"正在加载 Faster-Whisper ASR 模型: {model_size} (设备: {device}, 计算类型: {compute_type})")
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            logger.info("WhisperASR 初始化完毕。")
        except Exception as e:
            logger.error(f"加载 Faster-Whisper 模型失败: {e}")
            raise

    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        接收完整的音频数据并返回最终文本。
        参数 'audio_data' 应为 float32 的 numpy 数组，采样率为16kHz。
        """
        if not isinstance(audio_data, np.ndarray):
            logger.error("WhisperASR: 音频数据必须是 numpy 数组。")
            return ""

        if audio_data.dtype == np.int16:
            logger.debug("WhisperASR: 音频数据为 int16，将其转换为 float32。")
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype != np.float32:
            logger.warning(f"WhisperASR: 音频数据类型为 {audio_data.dtype}，期望 float32。尝试直接处理。")
            # 可以选择强制转换或抛出错误
            # audio_data = audio_data.astype(np.float32)

        if audio_data.ndim > 1:
            logger.debug(f"WhisperASR: 音频数据有 {audio_data.shape[1]} 个声道，将转换为单声道。")
            audio_data = np.mean(audio_data, axis=1) # 简单混合为单声道

        logger.info(f"WhisperASR: 开始转写音频数据，长度: {len(audio_data)} samples...")

        try:
            # language参数可以根据需要从opt获取或自动检测
            # beam_size, vad_filter等参数也可以从opt获取以增加灵活性
            segments, info = self.model.transcribe(audio_data,
                                                 beam_size=getattr(self.opt, 'whisper_beam_size', 5),
                                                 language=getattr(self.opt, 'whisper_language', None), # None表示自动检测
                                                 vad_filter=getattr(self.opt, 'whisper_vad_filter', True),
                                                 vad_parameters=dict(min_silence_duration_ms=500) # 可配置
                                                 )

            full_text = "".join([s.text for s in segments])
            logger.info(f"WhisperASR: 转写完成 - 检测到的语言 '{info.language}' (置信度 {info.language_probability:.2f})")
            logger.info(f"WhisperASR: 转写结果 - '{full_text}'")
            return full_text
        except Exception as e:
            logger.error(f"WhisperASR: 转写过程中发生错误: {e}")
            return ""

    # BaseASR中似乎没有定义抽象的run_step或类似的方法需要在这里实现
    # 如果需要处理流式ASR，则需要重写 run_step 和 get_next_feat，
    # 但根据您的PPT，transcribe接收完整音频，所以当前实现符合要求。
    # def run_step(self):
    #     # 用于流式处理的逻辑，当前方案用不到
    #     pass

    # def get_next_feat(self,block,timeout):
    #     # 用于流式处理的逻辑，当前方案用不到
    #     return self.feat_queue.get(block,timeout)

# 可选：添加一个简单的测试函数
if __name__ == '__main__':
    # 这是一个非常简单的测试，实际使用中opt需要由app.py提供
    class MockOpt:
        def __init__(self):
            self.whisper_model_size = 'tiny' # 使用小模型测试，避免下载大模型
            self.whisper_beam_size = 5
            self.whisper_language = 'en' # 指定语言避免下载检测模型
            self.whisper_vad_filter = True
            # BaseASR 所需参数
            self.fps = 25
            self.batch_size = 1
            self.l = 0 # stride_left_size
            self.r = 0 # stride_right_size


    print("测试 WhisperASR...")
    mock_opt = MockOpt()

    try:
        asr_instance = WhisperASR(mock_opt)

        # 创建一个假的音频数据 (1秒的静音，16kHz, float32)
        sample_rate = 16000
        duration = 1
        dummy_audio = np.zeros(sample_rate * duration, dtype=np.float32)

        # 尝试转写
        print("尝试转写静音音频...")
        text_output = asr_instance.transcribe(dummy_audio)
        print(f"转写结果: '{text_output}' (预期为空或类似)")

        # 你可以替换为实际的音频文件进行测试
        # import soundfile as sf
        # try:
        #     audio_data, sr = sf.read("path_to_your_audio.wav", dtype='float32')
        #     if sr != 16000:
        #         # 需要重采样到16kHz
        #         print(f"Warning: Audio sample rate is {sr}, Whisper expects 16kHz.")
        #     print("尝试转写实际音频文件...")
        #     text_output = asr_instance.transcribe(audio_data)
        #     print(f"转写结果: '{text_output}'")
        # except Exception as e:
        #     print(f"读取或转写音频文件失败: {e}")

    except Exception as e:
        print(f"WhisperASR 测试失败: {e}")
        print("请确保已安装 faster-whisper 并且模型可以下载。")
        print("pip install faster-whisper")

    print("WhisperASR 测试结束。")
