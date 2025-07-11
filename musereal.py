# musereal.py
# 继承BaseReal，实现流式MuseTalk驱动

import os
import cv2
import torch
import numpy as np
import time
import wave
import io
from PIL import Image

from basereal import BaseReal # 确保从项目根目录正确导入
from logger import logger # 使用项目统一的logger

# --- MuseTalk 相关导入 ---
# 确保这些导入路径相对于项目根目录是正确的
# 或者 MuseTalk 已作为包安装
try:
    from musetalk.models.unet_2d_condition import UNet2DConditionModel
    from musetalk.models.controlnet import ControlNetModel
    from musetalk.models.vae import AutoencoderKL
    from musetalk.pipelines.pipeline_controlnet_muse import MuseControlNetPipeline
    from transformers import CLIPImageProcessor
    from diffusers.schedulers import DDIMScheduler #, EulerDiscreteScheduler
    from musetalk.utils.scheduler_utils import get_scheduler
    from musetalk.utils.image_utils import get_video_from_images, resize_image
    from musetalk.whisper.audio2feature import Audio2Feature # MuseTalk自带的音频特征提取
    from musetalk.utils.utils import get_config
except ImportError as e:
    logger.error(f"MuseTalk 相关模块导入失败: {e}")
    logger.error("请确保 MuseTalk 已正确安装并且其路径在 PYTHONPATH 中，或者相关代码在工作目录下。")
    # 可以选择抛出异常或设置一个标志，使 MuseReal 不可用
    raise

class MuseReal(BaseReal):
    def __init__(self, opt):
        super().__init__(opt) # 调用父类的构造函数
        self.opt = opt
        self.is_initialized = False

        logger.info("正在初始化 MuseTalk 驱动...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"MuseTalk 将使用设备: {self.device}")

        # 从 opt 获取 MuseTalk 相关配置
        self.musetalk_model_dir = getattr(self.opt, 'musetalk_model_dir', './models/musetalk_hf')
        self.musetalk_config_path = getattr(self.opt, 'musetalk_config_path',
                                            os.path.join(self.musetalk_model_dir, 'musetalk/configuration.json')) # 默认路径可能需要调整
        self.musetalk_source_image_path = getattr(self.opt, 'musetalk_source_image', './data/avatars/musetalk_default.png')

        if not os.path.exists(self.musetalk_model_dir):
            logger.error(f"MuseTalk 模型目录不存在: {self.musetalk_model_dir}")
            return
        if not os.path.exists(self.musetalk_config_path):
            logger.error(f"MuseTalk 配置文件不存在: {self.musetalk_config_path}")
            return
        if not os.path.exists(self.musetalk_source_image_path):
            logger.error(f"MuseTalk 源图片不存在: {self.musetalk_source_image_path}")
            return

        logger.info(f"MuseTalk 模型目录: {self.musetalk_model_dir}")
        logger.info(f"MuseTalk 配置文件: {self.musetalk_config_path}")
        logger.info(f"MuseTalk 源图片: {self.musetalk_source_image_path}")

        try:
            self.cfg = get_config(self.musetalk_config_path)
            if self.cfg is None:
                logger.error(f"无法从 {self.musetalk_config_path} 加载配置。")
                return

            # 加载所有子模型
            logger.info("正在加载 MuseTalk VAE...")
            self.vae = AutoencoderKL.from_pretrained(os.path.join(self.musetalk_model_dir, self.cfg.model_config.vae_path)).to(self.device)

            logger.info("正在加载 MuseTalk UNet...")
            self.unet = UNet2DConditionModel.from_pretrained(os.path.join(self.musetalk_model_dir, self.cfg.model_config.unet_path), torch_dtype=torch.float16).to(self.device)

            logger.info("正在加载 MuseTalk ControlNet...")
            self.controlnet = ControlNetModel.from_pretrained(os.path.join(self.musetalk_model_dir, self.cfg.model_config.controlnet_path), torch_dtype=torch.float16).to(self.device)
            
            logger.info("正在加载 MuseTalk ImageProcessor...")
            self.image_processor = CLIPImageProcessor.from_pretrained(os.path.join(self.musetalk_model_dir, self.cfg.model_config.image_encoder_path))

            logger.info("正在加载 MuseTalk Scheduler...")
            # 使用 EulerDiscreteScheduler 作为示例，如果DDIMScheduler有问题
            # self.scheduler = EulerDiscreteScheduler.from_pretrained(os.path.join(self.musetalk_model_dir, self.cfg.model_config.scheduler_path))
            self.scheduler = get_scheduler(self.cfg.model_config.scheduler_path, DDIMScheduler)


            logger.info("正在创建 MuseTalk Pipeline...")
            self.pipeline = MuseControlNetPipeline(
                vae=self.vae, unet=self.unet, controlnet=self.controlnet,
                scheduler=self.scheduler, image_processor=self.image_processor
            ).to(self.device)

            # 加载并处理源图片
            logger.info(f"正在加载并处理源图片: {self.musetalk_source_image_path}")
            source_image_pil = Image.open(self.musetalk_source_image_path).convert("RGB")
            self.source_image_resized = resize_image(source_image_pil, self.cfg.video_config.size)

            # 初始化音频处理器 (用于将音频转换为梅尔频谱等特征)
            logger.info("正在初始化 MuseTalk 音频处理器...")
            self.audio_processor = Audio2Feature(self.cfg.audio_config, self.device)

            self.is_initialized = True
            logger.info("MuseTalk 驱动初始化完毕。")

        except Exception as e:
            logger.error(f"MuseTalk 模型加载或初始化失败: {e}", exc_info=True)
            # self.is_initialized 保持 False

    def _run(self):
        """
        驱动器主循环，从音频队列获取数据并生成视频帧。
        这个方法会在 BaseReal 的 start() 方法中被一个新线程调用。
        """
        if not self.is_initialized:
            logger.error("MuseReal 未成功初始化，无法运行。")
            return

        logger.info("MuseReal _run 循环启动。等待音频数据...")
        while not self.closed: # self.closed 是从 BaseReal 继承的，用于优雅关闭
            if self.audio_queue.empty():
                time.sleep(0.01) # 短暂休眠避免CPU空转
                continue

            # self.get_audio_chunk() 是 BaseReal 提供的方法，用于从 self.audio_queue 获取数据
            # 它应该返回 bytes 类型的音频数据块
            audio_data_bytes = self.get_audio_chunk()
            if audio_data_bytes is None: # 可能表示队列结束信号或其他
                logger.info("MuseReal: 从音频队列获取到 None，可能准备关闭。")
                continue

            if not audio_data_bytes: # 空的 bytes 对象
                logger.debug("MuseReal: 获取到空的音频数据块，跳过。")
                continue

            logger.info(f"MuseReal: 接收到 {len(audio_data_bytes)}字节 音频块，开始生成视频帧...")

            try:
                # 1. 将音频字节流转换为 MuseTalk 需要的格式 (通常是 float32 numpy array)
                #    假设音频是16kHz, 单声道, 16-bit PCM (WAV格式常见)
                with io.BytesIO(audio_data_bytes) as audio_buffer:
                    with wave.open(audio_buffer, 'rb') as wf:
                        n_channels = wf.getnchannels()
                        sampwidth = wf.getsampwidth()
                        framerate = wf.getframerate()
                        n_frames = wf.getnframes()
                        audio_frames_bytes = wf.readframes(n_frames)

                        if framerate != self.cfg.audio_config.sample_rate: # MuseTalk 通常是16000
                             logger.warning(f"输入音频采样率 ({framerate}Hz) 与 MuseTalk 配置 ({self.cfg.audio_config.sample_rate}Hz) 不符。请确保TTS输出正确采样率的音频。")
                             # 此处可能需要重采样，但更推荐TTS直接输出正确格式

                        # 根据声道数和位深转换为numpy数组
                        if sampwidth == 2: # 16-bit
                            audio_array_int16 = np.frombuffer(audio_frames_bytes, dtype=np.int16)
                        elif sampwidth == 1: # 8-bit
                            audio_array_int16 = (np.frombuffer(audio_frames_bytes, dtype=np.uint8).astype(np.int16) - 128) * 256
                        else:
                            logger.error(f"不支持的音频位深: {sampwidth}")
                            continue

                        # 如果是多声道，混合为单声道 (简单平均)
                        if n_channels > 1:
                            audio_array_int16 = audio_array_int16.reshape(-1, n_channels).mean(axis=1).astype(np.int16)

                # 将 int16 numpy 数组转换为 float32，并归一化到 [-1, 1]
                audio_array_float32 = audio_array_int16.astype(np.float32) / 32768.0

                # 2. 使用 MuseTalk 的 Audio2Feature 提取梅尔频谱等特征
                #    get_mel_chunks_from_array 需要 (audio_array, sample_rate, fps)
                #    fps 应该是目标视频的fps，例如 cfg.video_config.fps
                audio_mel_chunks = self.audio_processor.get_mel_chunks_from_array(
                    audio_array_float32,
                    self.cfg.audio_config.sample_rate, # 确保使用配置的采样率
                    self.cfg.video_config.fps
                )

                if not audio_mel_chunks or len(audio_mel_chunks) == 0:
                    logger.info("MuseReal: 从音频数据中未提取到有效的梅尔频谱块。")
                    continue

                # 3. 执行核心推理管线，逐帧生成
                #    pipeline的参数根据 MuseTalk 的具体实现调整
                logger.info(f"MuseReal: 正在使用 {len(audio_mel_chunks)} 个梅尔频谱块进行推理...")
                pipeline_output = self.pipeline(
                    image = self.source_image_resized, # PIL Image
                    audio_conds = audio_mel_chunks,   # List of mel chunks
                    # --- 以下参数来自PPT中的cfg.pipeline_config ---
                    # 这些参数可能需要根据你的 musetalk configuration.json 调整
                    # 或者直接解包 self.cfg.pipeline_config (如果适用)
                    # 例如:
                    #   width=self.cfg.video_config.size[0],
                    #   height=self.cfg.video_config.size[1],
                    #   num_inference_steps=self.cfg.pipeline_config.get('num_inference_steps', 10),
                    #   guidance_scale=self.cfg.pipeline_config.get('guidance_scale', 3.5),
                    #   controlnet_conditioning_scale=float(self.cfg.pipeline_config.get('controlnet_conditioning_scale', 0.8)),
                    #   output_type="pil", # "pil" or "np"
                    #   num_frames_per_cond=1, # 通常是1
                    **self.cfg.pipeline_config # 直接传递配置字典
                )
                result_frames_pil = pipeline_output.frames # 假设输出是PIL Image列表

                logger.info(f"MuseReal: 成功生成 {len(result_frames_pil)} 帧视频。")

                # 4. 将生成的视频帧 (PIL Image) 转换为 BGR np.ndarray 并推送到WebRTC轨道
                for frame_pil in result_frames_pil:
                    # 将 PIL Image 转换为 OpenCV BGR格式的 NumPy 数组
                    frame_rgb = np.array(frame_pil)
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                    # self.push_frame 是 BaseReal 提供的方法，用于将帧推送到绑定的WebRTC轨道
                    # 它期望 (frame_bgr_numpy_array, eventpoint)
                    # eventpoint可以用于同步，这里我们暂时不传递复杂的eventpoint
                    self.push_frame(frame_bgr, None)
                    # logger.debug("MuseReal: 推送一帧到 WebRTC。")

            except wave.Error as e:
                logger.error(f"MuseReal: 处理WAV音频数据失败: {e}. 音频数据可能不是有效的WAV格式。")
            except Exception as e:
                logger.error(f"MuseReal: 视频帧生成或推送过程中发生严重错误: {e}", exc_info=True)
        
        logger.info("MuseReal _run 循环结束。")

    # BaseReal的receive方法会将音频数据放入self.audio_queue，_run方法会消费它
    # BaseReal的start方法会启动_run在一个新线程中
    # BaseReal的process_frames方法（如果被调用）会从res_frame_queue（如果MuseReal填充它）中获取帧
    # 但在此设计中，_run直接调用self.push_frame，所以不需要res_frame_queue或重写process_frames

    # __del__ 方法可以保留，用于清理（如果需要）
    def __del__(self):
        logger.info(f"MuseReal({self.sessionid}) 实例正在被销毁。")
        # 可以在这里添加任何必要的清理代码，例如释放GPU资源
        # 但通常模型和pipeline在Python对象销毁时会自动处理
        if hasattr(self, 'pipeline') and hasattr(self.pipeline, 'to'):
            try:
                # 尝试将模型移回CPU，以防万一有助于显存释放
                # self.pipeline.to('cpu')
                # del self.pipeline
                # del self.vae
                # del self.unet
                # del self.controlnet
                # if torch.cuda.is_available():
                #    torch.cuda.empty_cache()
                logger.debug("MuseReal: 尝试清理模型资源。")
            except Exception as e:
                logger.warning(f"MuseReal: 清理模型资源时出错: {e}")

# 用于独立测试的简单桩代码 (可选)
if __name__ == '__main__':
    logger.info("测试 MuseReal 模块...")

    # 创建一个模拟的opt对象
    class MockOpt:
        def __init__(self):
            self.sessionid = "test_session"
            self.fps = 25 # BaseReal 需要
            self.tts = "edgetts" # BaseReal 需要，但这里不实际使用TTS
            self.customopt = [] # BaseReal 需要

            # MuseTalk 特定参数 - 根据你的实际文件路径修改
            self.musetalk_model_dir = os.getenv("MUSETALK_MODEL_DIR", "../../models/musetalk_hf") # 假设模型在项目根目录的models下
            self.musetalk_config_path = os.getenv("MUSETALK_CONFIG_PATH", os.path.join(self.musetalk_model_dir, "musetalk/configuration.json"))
            self.musetalk_source_image = os.getenv("MUSETALK_SOURCE_IMAGE", "../../data/avatars/musetalk_default.png") # 假设图片在项目根目录的data下

            # 确保路径是相对于当前脚本的，或者使用绝对路径
            # 如果脚本在项目根目录，上面的相对路径可能需要调整
            # 例如，如果此脚本在 LiveTalking/ 目录下，而模型在 LiveTalking/models/musetalk_hf
            # 则路径应为 './models/musetalk_hf'

            # 修正路径使其相对于当前文件musereal.py的父目录（即项目根目录）
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir) # 假设musereal.py在项目根目录下的一个子目录里 (实际上它在根目录)
                                                        # 如果musereal.py就在项目根目录，project_root = current_dir

            # 假设 musereal.py 就在项目根目录
            project_root = current_dir

            self.musetalk_model_dir = os.path.join(project_root, "models/musetalk_hf")
            self.musetalk_config_path = os.path.join(self.musetalk_model_dir, "musetalk/configuration.json")
            self.musetalk_source_image = os.path.join(project_root, "data/avatars/musetalk_default.png")

            logger.info(f"MockOpt: musetalk_model_dir='{self.musetalk_model_dir}'")
            logger.info(f"MockOpt: musetalk_config_path='{self.musetalk_config_path}'")
            logger.info(f"MockOpt: musetalk_source_image='{self.musetalk_source_image}'")


    mock_opt = MockOpt()
    
    # 检查依赖文件是否存在
    if not os.path.exists(mock_opt.musetalk_model_dir):
        logger.error(f"测试错误: MuseTalk 模型目录不存在: {mock_opt.musetalk_model_dir}")
        exit()
    if not os.path.exists(mock_opt.musetalk_config_path):
        logger.error(f"测试错误: MuseTalk 配置文件不存在: {mock_opt.musetalk_config_path}")
        exit()
    if not os.path.exists(mock_opt.musetalk_source_image):
        logger.error(f"测试错误: MuseTalk 源图片不存在: {mock_opt.musetalk_source_image}")
        exit()

    try:
        muse_real_instance = MuseReal(mock_opt)

        if muse_real_instance.is_initialized:
            logger.info("MuseReal 实例初始化成功。")
            
            # 模拟音频数据 (例如，1秒的16kHz, 16-bit PCM mono WAV)
            sample_rate = 16000
            duration_s = 1
            frequency = 440 # A4音
            t = np.linspace(0, duration_s, int(sample_rate * duration_s), False)
            audio_signal_float = 0.5 * np.sin(2 * np.pi * frequency * t)
            audio_signal_int16 = (audio_signal_float * 32767).astype(np.int16)

            # 创建WAV字节流
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2) # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio_signal_int16.tobytes())
            wav_bytes = wav_buffer.getvalue()

            # 模拟 BaseReal 的 receive 方法将音频放入队列
            logger.info("模拟将音频数据放入队列...")
            muse_real_instance.audio_queue.put(wav_bytes)

            # 启动 _run 循环 (在测试中直接调用，实际由BaseReal.start()在新线程中调用)
            # 为了测试，我们让它运行一小段时间然后关闭
            muse_real_instance.closed = False # 确保未关闭

            # 由于 _run 是一个循环，我们需要一个方式来停止它
            # 在真实场景中，BaseReal的stop()方法会设置self.closed = True
            
            # 这里我们不能直接调用 _run() 因为它会阻塞
            # 我们可以启动它在一个线程中，然后模拟接收帧

            # 简单的做法：直接调用一次核心逻辑，而不是整个循环
            def single_pass_test():
                if not muse_real_instance.audio_queue.empty():
                    audio_data_bytes_test = muse_real_instance.get_audio_chunk()
                    if audio_data_bytes_test:
                        # ... (复制 _run 内部的 try-except 块逻辑) ...
                        # 这部分比较复杂，暂时跳过直接执行pipeline，只确认音频能被取出
                        logger.info(f"单次测试：成功从队列取出 {len(audio_data_bytes_test)} 字节音频。")
                        logger.info("要完整测试 _run，需要更复杂的模拟环境或启动线程。")
                else:
                    logger.info("单次测试：音频队列为空。")

            # single_pass_test()

            logger.info("MuseReal 模块基本测试完成。要进行完整功能测试，请在 LiveTalking 应用中运行。")
            # 在实际应用中，BaseReal.start()会启动_run线程
            # BaseReal.push_frame 会将帧发送到WebRTC
            # 此处仅测试初始化和基本结构

        else:
            logger.error("MuseReal 实例初始化失败。请检查日志中的错误信息。")

    except Exception as e:
        logger.error(f"MuseReal 测试过程中发生未知错误: {e}", exc_info=True)

    finally:
        # 清理，例如关闭可能由 MuseReal 打开的资源（如果它管理的话）
        if 'muse_real_instance' in locals() and hasattr(muse_real_instance, 'close'):
             muse_real_instance.close() # 假设BaseReal有close方法
        logger.info("MuseReal 测试结束。")
