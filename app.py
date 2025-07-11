# app.py
# LiveTalking 主控程序 - 修改版，集成 ASR/LLM/MuseTalk

import base64
import json
import re
import numpy as np
from threading import Thread,Event
import torch.multiprocessing as mp

from aiohttp import web
import aiohttp
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription,RTCIceServer,RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender
# from webrtc import HumanPlayer # HumanPlayer 将在下面定义，因为它需要访问 nerfreals
from basereal import BaseReal
# from llm import llm_response # 将被新的 OllamaLLM 替代

import argparse
import random
import shutil
import asyncio
import torch
from typing import Dict
from logger import logger
import gc
import soundfile as sf
import io

# --- 新模块导入 ---
from whisper_asr import WhisperASR
from ollama_llm import OllamaLLM
# TTS 引擎从 BaseReal 内部或 ttsreal.py 获取
from ttsreal import TTSEngine # 假设 TTSEngine 在 ttsreal.py 中定义并可独立使用

# --- 数字人驱动器导入 ---
from musereal import MuseReal
from lipreal import LipReal # 保留，以备切换
# from ernerf_real import ErnerfReal
# from ultralight_real import UltralightReal

# --- 全局变量和应用上下文 ---
# opt 将在 main 函数中被赋值
opt_global = None
asr_instance_global = None
llm_instance_global = None
tts_engine_global = None # TTS引擎现在独立于 BaseReal 实例

# nerfreals 存储 <sessionid, BaseReal 实例> 的映射
nerfreals: Dict[int, BaseReal] = {}
pcs = set() # 存储 RTCPeerConnection 实例

# --- HumanPlayer 定义 (从 webrtc.py 移入并调整) ---
class HumanPlayer:
    def __init__(self, sessionid, nerfreal_instance: BaseReal):
        self.sessionid = sessionid
        self.nerfreal = nerfreal_instance # BaseReal instance
        
        # BaseReal 实例应该有一个方法来提供音频和视频轨道
        # 或者，BaseReal 应该有方法来注册 aiortc 的 AudioStreamTrack 和 VideoStreamTrack
        # 从现有代码看，BaseReal.render 会启动 process_frames,
        # process_frames 会将数据放入 audio_track._queue 和 video_track._queue
        # 我们需要确保这些 track 被正确创建并传递给 BaseReal

        self.audio_track = NerfAudioTrack(self.nerfreal)
        self.video_track = NerfVideoTrack(self.nerfreal)

        # 启动 BaseReal 的渲染循环，它会开始填充上述 track 的队列
        # 这个启动逻辑现在移到创建 BaseReal 实例之后
        # self.nerfreal.start()

    @property
    def audio(self):
        return self.audio_track

    @property
    def video(self):
        return self.video_track

# --- aiortc Track 定义 (模拟 webrtc.py 中的 NerfAudioTrack/VideoTrack) ---
# 这些类需要能够从 BaseReal 的队列中获取数据
class NerfTrack: # 基类
    def __init__(self, real_instance: BaseReal):
        self.real_instance = real_instance
        self._queue = asyncio.Queue() # aiortc 的 track 通常使用 asyncio.Queue
        self.real_instance.register_track_queues(self.kind, self._queue) # BaseReal需要这个方法

    async def recv(self):
        # logger.debug(f"{self.kind} track recv called, queue size: {self._queue.qsize()}")
        frame, eventpoint = await self._queue.get()
        # logger.debug(f"{self.kind} track got frame, event: {eventpoint}")
        if eventpoint is not None and hasattr(self.real_instance, 'notify'):
            self.real_instance.notify(eventpoint) # 通知事件点
        return frame

class NerfAudioTrack(NerfTrack, aiortc.mediastreams.MediaStreamTrack):
    kind = "audio"
    def __init__(self, real_instance: BaseReal):
        super().__init__(real_instance)
        # NerfTrack.__init__(self, real_instance) # MRO issue, call explicitly if needed

class NerfVideoTrack(NerfTrack, aiortc.mediastreams.MediaStreamTrack):
    kind = "video"
    def __init__(self, real_instance: BaseReal):
        super().__init__(real_instance)
        # NerfTrack.__init__(self, real_instance)

# --- BaseReal 需要一个新方法来注册这些队列 ---
# 我们需要在 BaseReal 定义中添加:
# def register_track_queues(self, kind, queue_instance):
#     if kind == "audio":
#         self.aio_audio_track_queue = queue_instance
#     elif kind == "video":
#         self.aio_video_track_queue = queue_instance
#
# 并在 process_frames 中使用这些 asyncio.Queue:
# (修改basereal.py)
#  asyncio.run_coroutine_threadsafe(video_track._queue.put((new_frame,None)), loop)
#  -> (假设 loop 和 track._queue 是 self.aio_video_track_queue)
#  asyncio.run_coroutine_threadsafe(self.aio_video_track_queue.put((new_frame,None)), self.event_loop_for_tracks)
# 这部分修改比较复杂，暂时假设 BaseReal.process_frames 内部能正确处理 aiortc.MediaStreamTrack 的 _queue

def randN(N)->int:
    min_val = pow(10, N - 1)
    max_val = pow(10, N)
    return random.randint(min_val, max_val - 1)

def build_nerfreal_instance(sessionid:int, current_opt) -> BaseReal:
    """
    构建并返回一个 BaseReal (数字人驱动器) 实例。
    不再依赖全局 model 和 avatar。
    """
    current_opt.sessionid = sessionid # 为当前会话设置ID
    
    logger.info(f"构建数字人驱动器，模型: {current_opt.model}, 会话ID: {sessionid}")

    real_instance = None
    if current_opt.model == 'musetalk':
        # MuseReal 的 __init__ 现在只接收 opt
        real_instance = MuseReal(current_opt)
    elif current_opt.model == 'wav2lip':
        # LipReal 的 __init__ 也需要修改为只接收 opt，并在内部加载模型
        # 为了演示，我们假设 LipReal 也遵循类似 MuseReal 的模式
        # LipReal(opt, model, avatar) -> LipReal(opt)
        # 这需要对 lipreal.py 进行与 musereal.py 类似的修改
        logger.warning("Wav2Lip (LipReal) 的模型加载逻辑可能需要更新为从 opt 初始化。")
        # 临时的兼容代码，如果 lipreal.py 未更新:
        try:
            from lipreal import load_model as lip_load_model, load_avatar as lip_load_avatar
            model_lip = lip_load_model("./models/wav2lip.pth") # 路径应可配置
            avatar_lip = lip_load_avatar(current_opt.avatar_id) # avatar_id 需为 wav2lip 定义
            real_instance = LipReal(current_opt, model_lip, avatar_lip)
        except Exception as e:
            logger.error(f"加载旧版 LipReal 失败: {e}. 请更新 LipReal 以从 opt 初始化。")
            raise # 或者返回 None

    # ... 其他模型的加载逻辑 ...
    else:
        logger.error(f"不支持的模型类型: {current_opt.model}")
        raise ValueError(f"不支持的模型类型: {current_opt.model}")

    if real_instance and hasattr(real_instance, 'is_initialized') and not real_instance.is_initialized:
        logger.error(f"{current_opt.model} 实例初始化失败。请检查日志。")
        raise RuntimeError(f"{current_opt.model} 实例初始化失败。")

    if real_instance:
         # 获取当前事件循环，用于在 BaseReal 的 process_frames 中安全地向aiortc轨道队列put数据
        try:
            loop = asyncio.get_event_loop()
            real_instance.set_event_loop(loop) # BaseReal需要一个set_event_loop方法
        except RuntimeError: # 可能在非主线程中没有当前循环
            logger.warning("无法在 build_nerfreal_instance 中获取当前事件循环。")
            # real_instance.set_event_loop(asyncio.new_event_loop()) # 或者传递一个固定的loop

        real_instance.start() # 启动 BaseReal 的后台线程 (_run, process_frames)
        logger.info(f"{current_opt.model} 实例 (ID: {sessionid}) 已启动。")

    return real_instance

# --- WebRTC 信令端点 (基本保持不变，但实例化 Player 的方式会变) ---
async def offer_handler(request): # Renamed from 'offer' to avoid conflict
    params = await request.json()
    offer_desc = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    sessionid = randN(6)
    logger.info(f"新会话请求，生成ID: {sessionid}, 当前会话数: {len(nerfreals)}")

    try:
        # 使用全局的 opt_global 来构建实例
        # build_nerfreal_instance 现在会启动 real_instance 的线程
        loop = asyncio.get_event_loop()
        real_instance = await loop.run_in_executor(None, build_nerfreal_instance, sessionid, opt_global)
        if not real_instance:
             raise RuntimeError("数字人实例创建失败")
        nerfreals[sessionid] = real_instance
    except Exception as e:
        logger.error(f"创建数字人实例失败 (会话ID {sessionid}): {e}", exc_info=True)
        return web.Response(
            content_type="application/json",
            text=json.dumps({"error": f"Failed to create digital human instance: {str(e)}"}),
            status=500
        )

    ice_server = RTCIceServer(urls=getattr(opt_global, 'stun_server', 'stun:stun.l.google.com:19302'))
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[ice_server]))
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"会话 {sessionid}: 连接状态变为 {pc.connectionState}")
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            logger.info(f"会话 {sessionid}: 连接关闭或失败。正在清理资源...")
            await pc.close()
            pcs.discard(pc)
            if sessionid in nerfreals:
                try:
                    nerfreals[sessionid].stop() # BaseReal 需要一个 stop 方法来清理线程和资源
                except Exception as e:
                    logger.error(f"关闭 BaseReal 实例 (ID: {sessionid}) 时出错: {e}")
                del nerfreals[sessionid]
                logger.info(f"会话 {sessionid}: 资源已清理。剩余会话数: {len(nerfreals)}")
            gc.collect() # 强制垃圾回收

    # HumanPlayer 现在接收 sessionid 和 real_instance
    player = HumanPlayer(sessionid, nerfreals[sessionid])
    if player.audio:
        pc.addTrack(player.audio)
    if player.video:
        pc.addTrack(player.video)

    # Codec preferences (保持不变)
    # capabilities = RTCRtpSender.getCapabilities("video")
    # preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs)) # H264 often preferred
    # preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
    # preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs)) # Keep rtx for retransmissions
    # if pc.getTransceivers() and len(pc.getTransceivers()) > 1:
    #     transceiver = pc.getTransceivers()[1] # Assuming video is the second transceiver
    #     if transceiver and hasattr(transceiver, 'setCodecPreferences'):
    #        transceiver.setCodecPreferences(preferences)
    # else:
    #     logger.warning(f"会话 {sessionid}: 无法设置编解码器偏好，可能没有视频轨道或收发器。")


    await pc.setRemoteDescription(offer_desc)
    answer_desc = await pc.createAnswer()
    await pc.setLocalDescription(answer_desc)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid": sessionid}
        ),
    )

# --- 新的HTTP端点，用于处理音频输入并驱动整个AI流程 ---
async def process_audio_handler(request):
    if not opt_global or not asr_instance_global or not llm_instance_global or not tts_engine_global:
        logger.error("全局模块 (opt, ASR, LLM, TTS) 未初始化。")
        return web.Response(
            content_type="application/json",
            text=json.dumps({"error": "服务器模块未完全初始化。"}),
            status=503 # Service Unavailable
        )
        
    try:
        data = await request.json() # 假设前端发送JSON: {"sessionid": id, "audio_base64": "..."}
                                    # 或者使用 multipart/form-data for file upload
                                    # 为了简单，这里用 base64 编码的 WAV 数据
        sessionid = data.get('sessionid')
        audio_base64 = data.get('audio_data_base64') # 客户端发送完整的WAV文件的Base64编码

        if not sessionid or audio_base64 is None:
            return web.Response(
                content_type="application/json",
                text=json.dumps({"error": "缺少 sessionid 或 audio_data_base64"}),
                status=400)

        sessionid = int(sessionid)
        if sessionid not in nerfreals:
            return web.Response(
                content_type="application/json",
                text=json.dumps({"error": f"无效的 sessionid: {sessionid}"}),
                status=404)

        logger.info(f"会话 {sessionid}: 接收到音频数据，开始处理AI流程...")

        # 1. 解码 Base64 音频数据并转换为 NumPy 数组
        try:
            audio_bytes = base64.b64decode(audio_base64)
            with io.BytesIO(audio_bytes) as audio_buffer:
                # 确保 soundfile 可以处理内存中的字节流
                audio_np, sample_rate = sf.read(audio_buffer, dtype='float32')
            logger.info(f"会话 {sessionid}: 音频解码成功，采样率: {sample_rate}, 形状: {audio_np.shape}")
            if sample_rate != 16000: # Whisper 通常期望16kHz
                logger.warning(f"会话 {sessionid}: 音频采样率为 {sample_rate}，ASR可能需要16kHz。请确保客户端发送16kHz音频。")
                # 此处可以添加重采样逻辑，但最好由客户端保证
        except Exception as e:
            logger.error(f"会话 {sessionid}: 解码或读取音频数据失败: {e}", exc_info=True)
            return web.Response(
                content_type="application/json",
                text=json.dumps({"error": f"无效的音频数据: {str(e)}"}),
                status=400)

        # 2. ASR 转写
        logger.info(f"会话 {sessionid}: 开始ASR转写...")
        asr_text = asr_instance_global.transcribe(audio_np)
        if not asr_text:
            logger.info(f"会话 {sessionid}: ASR未返回文本。流程中止。")
            return web.Response(
                content_type="application/json",
                text=json.dumps({"message": "ASR未能识别语音。", "asr_text": ""}),
                status=200) # 可能是静音或无法识别，不一定是错误

        logger.info(f"会话 {sessionid}: ASR结果: '{asr_text}'")

        # (可选) 视觉描述逻辑 - 此处简化，暂不实现
        # image_path = "path/to/current/scene/image.png" # 需要一种方式获取当前视觉场景
        # visual_prompt = "描述这张图片中的主要内容。"
        # visual_description = llm_instance_global.get_visual_description(image_path, visual_prompt)
        # combined_prompt_for_llm = f"用户说：'{asr_text}'。当前场景描述：'{visual_description}'。请回复用户。"

        # 3. LLM 处理 (流式)
        logger.info(f"会话 {sessionid}: LLM开始处理文本: '{asr_text}'")

        # 简单的标点符号列表，用于分割LLM的回复以进行TTS
        # 可以从 opt_global 获取更复杂的分割逻辑或字符列表
        sentence_terminators = getattr(opt_global, 'sentence_terminators', "。？！，,!?;")

        llm_response_stream = llm_instance_global.chat_stream(asr_text) # history可以后续加入

        current_sentence_chunk = ""
        full_llm_response_for_log = []

        async for text_piece in llm_response_stream: # chat_stream现在是异步生成器
            if text_piece is None: continue # 以防万一生成器产生None

            current_sentence_chunk += text_piece
            full_llm_response_for_log.append(text_piece)

            # 检查是否应该将当前块发送到TTS
            # （例如，遇到标点符号，或者达到一定长度）
            should_send_to_tts = False
            if any(p in text_piece for p in sentence_terminators):
                 should_send_to_tts = True
            # 可以增加长度判断: or len(current_sentence_chunk) > getattr(opt_global, 'tts_chunk_length', 50)

            if should_send_to_tts and current_sentence_chunk.strip():
                text_to_synthesize = current_sentence_chunk.strip()
                logger.info(f"会话 {sessionid}: TTS处理文本块: '{text_to_synthesize}'")

                # 4. TTS 合成
                # tts_engine_global.synthesize 现在应该返回音频字节 (wav)
                tts_audio_bytes = await tts_engine_global.synthesize_async(text_to_synthesize) # 假设有异步版本
                if not tts_audio_bytes:
                    logger.warning(f"会话 {sessionid}: TTS未能从文本块 '{text_to_synthesize}' 生成音频。")
                else:
                    logger.info(f"会话 {sessionid}: TTS生成 {len(tts_audio_bytes)} 字节音频。")
                    # 5. 发送音频到数字人驱动器
                    if sessionid in nerfreals:
                        # BaseReal.receive() 应该处理字节数据并放入其内部队列
                        nerfreals[sessionid].receive(tts_audio_bytes)
                        logger.debug(f"会话 {sessionid}: TTS音频块已发送给数字人驱动器。")
                    else:
                        logger.warning(f"会G话 {sessionid}: 会话已不存在，无法发送TTS音频。")
                        break # 如果会话没了，后续处理也无意义
                current_sentence_chunk = "" # 重置当前块

        # 处理LLM流结束后剩余的文本
        if current_sentence_chunk.strip():
            logger.info(f"会话 {sessionid}: TTS处理最后文本块: '{current_sentence_chunk.strip()}'")
            tts_audio_bytes = await tts_engine_global.synthesize_async(current_sentence_chunk.strip())
            if tts_audio_bytes and sessionid in nerfreals:
                nerfreals[sessionid].receive(tts_audio_bytes)
                logger.debug(f"会话 {sessionid}: 最后的TTS音频块已发送给数字人驱动器。")

        logger.info(f"会话 {sessionid}: LLM完整回复: '{''.join(full_llm_response_for_log)}'")
        logger.info(f"会话 {sessionid}: AI流程处理完毕。")

        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "message": "音频处理成功。",
                "asr_text": asr_text,
                "llm_response_preview": "".join(full_llm_response_for_log)[:200] # 预览部分回复
            }),
            status=200
        )

    except Exception as e:
        logger.error(f"处理音频时发生严重错误: {e}", exc_info=True)
        sessionid_str = f"会话 {sessionid}" if 'sessionid' in locals() else "未知会话"
        return web.Response(
            content_type="application/json",
            text=json.dumps({"error": f"{sessionid_str}: 服务器内部错误 - {str(e)}"}),
            status=500
        )

# --- 其他辅助端点 (从旧代码中移植和调整) ---
async def human_text_input_handler(request): # 替换旧的 'human' 端点部分逻辑
    params = await request.json()
    sessionid = params.get('sessionid')
    text_input = params.get('text')
    interrupt = params.get('interrupt', False)

    if not sessionid or text_input is None:
        return web.Response(status=400, text="Missing sessionid or text")
    sessionid = int(sessionid)
    if sessionid not in nerfreals:
        return web.Response(status=404, text=f"Invalid sessionid: {sessionid}")

    if interrupt:
        nerfreals[sessionid].flush_talk() # BaseReal 需要这个方法

    # 此端点现在直接将文本送入TTS -> Avatar，绕过ASR和LLM
    # 这可以用于测试TTS和Avatar，或用于预设回复
    logger.info(f"会话 {sessionid}: 接收到直接文本输入 '{text_input}'，将送入TTS。")

    tts_audio_bytes = await tts_engine_global.synthesize_async(text_input)
    if tts_audio_bytes:
        nerfreals[sessionid].receive(tts_audio_bytes)
        return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "文本已发送至TTS和数字人"}))
    else:
        return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "TTS合成失败"}), status=500)

async def interrupt_talk_handler(request):
    params = await request.json()
    sessionid = params.get('sessionid')
    if sessionid is None: return web.Response(status=400, text="Missing sessionid")
    sessionid = int(sessionid)
    if sessionid not in nerfreals: return web.Response(status=404, text="Invalid sessionid")

    nerfreals[sessionid].flush_talk() # BaseReal 需要这个方法
    logger.info(f"会话 {sessionid}: 执行打断。")
    return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))

async def is_speaking_handler(request):
    params = await request.json()
    sessionid = params.get('sessionid')
    if sessionid is None: return web.Response(status=400, text="Missing sessionid")
    sessionid = int(sessionid)
    if sessionid not in nerfreals: return web.Response(status=404, text="Invalid sessionid")

    speaking_status = nerfreals[sessionid].is_speaking() # BaseReal 需要这个方法
    return web.Response(content_type="application/json", text=json.dumps({"code": 0, "data": speaking_status}))


async def on_aiohttp_shutdown(app_instance): # Renamed from on_shutdown
    logger.info("服务器正在关闭，清理 WebRTC 连接...")
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros, return_exceptions=True) # return_exceptions以防某个close失败
    pcs.clear()
    logger.info("所有 WebRTC 连接已关闭。")
    # 可以在这里添加其他清理逻辑，例如停止 BaseReal 实例
    for sid, real_instance in nerfreals.items():
        logger.info(f"正在停止数字人实例 (ID: {sid})...")
        if hasattr(real_instance, 'stop'):
            try:
                real_instance.stop()
            except Exception as e:
                 logger.error(f"关闭数字人实例 {sid} 时出错: {e}")
    nerfreals.clear()


# --- 主程序入口 ---
if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn', force=True)
    # 在Windows或macOS上，'spawn' 是更安全的选择，尤其是在使用CUDA时
    # 在Linux上，'fork' 通常是默认的，但在某些情况下 'spawn' 或 'forkserver' 可能更好
    try:
        mp.set_start_method('spawn')
        logger.info("多进程启动方式设置为 'spawn'。")
    except RuntimeError as e:
        logger.warning(f"设置多进程启动方式失败 (可能已设置): {e}")


    parser = argparse.ArgumentParser(description="LiveTalking Server with ASR/LLM/MuseTalk Integration")

    # --- 保留的原有参数 (部分可能需要调整或由新模块配置覆盖) ---
    parser.add_argument('--fps', type=int, default=25, help="Video FPS for avatar. BaseReal.chunk calculation depends on this.") # MuseTalk has its own FPS config
    # parser.add_argument('-l', type=int, default=10, help="ASR context window related, less relevant for Whisper full transcribe")
    # parser.add_argument('-m', type=int, default=8)
    # parser.add_argument('-r', type=int, default=10)
    parser.add_argument('--W', type=int, default=450, help="Default width for some old avatars, MuseTalk uses its own config.")
    parser.add_argument('--H', type=int, default=450, help="Default height for some old avatars.")
    # parser.add_argument('--batch_size', type=int, default=1, help="Batch size for some old avatar models.") # MuseTalk uses its own

    # TTS (这些参数现在由独立的TTSEngine使用，或通过opt传递给BaseReal内部的TTS)
    parser.add_argument('--tts', type=str, default='edgetts', help="TTS engine type (e.g., edgetts, xtts, gpt-sovits, etc.)")
    parser.add_argument('--REF_FILE', type=str, default="zh-CN-YunxiaNeural", help="Reference speaker/voice for TTS (e.g., EdgeTTS voice name, path to XTTS ref audio)")
    parser.add_argument('--REF_TEXT', type=str, default=None, help="Reference text for some TTS engines.")
    parser.add_argument('--TTS_SERVER', type=str, default='http://127.0.0.1:9880', help="URL for external TTS server (e.g., GPT-SoVITS)")
    parser.add_argument('--xtts_model_path', type=str, default='./models/xtts_v2', help="Path to XTTSv2 model directory.")
    parser.add_argument('--xtts_config_path', type=str, default='./models/xtts_v2/config.json', help="Path to XTTSv2 config file.")
    parser.add_argument('--xtts_speaker_wav', type=str, default="female_voice.wav", help="Path to XTTSv2 speaker wav for voice cloning.")


    # 数字人模型选择
    parser.add_argument('--model', type=str, default='musetalk', choices=['musetalk', 'wav2lip'], help='Digital human driver model type.')
    # Wav2Lip (如果使用)
    parser.add_argument('--avatar_id', type=str, default='default_avatar', help="Avatar ID for Wav2Lip (if used).")


    # WebRTC 和服务器
    parser.add_argument('--listen_port', type=int, default=8010, help="Web server listen port.")
    parser.add_argument('--stun_server', type=str, default='stun:stun.l.google.com:19302', help="STUN server for WebRTC.")
    # parser.add_argument('--max_session', type=int, default=1, help="Max concurrent WebRTC sessions (not strictly enforced in current version).")
    # parser.add_argument('--transport', type=str, default='webrtc', help="Transport type (webrtc, rtcpush, virtualcam - current focus is webrtc).")
    # parser.add_argument('--push_url', type=str, default='http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream', help="RTC push URL if using rtcpush.")

    # --- 新增的参数 ---
    # ASR (WhisperASR)
    parser.add_argument('--whisper_model_size', type=str, default='base', help='Faster-Whisper model size (e.g., tiny, base, small, medium, large-v2).')
    parser.add_argument('--whisper_language', type=str, default=None, help='Whisper language code (e.g., en, zh). None for auto-detect.')
    parser.add_argument('--whisper_beam_size', type=int, default=5, help='Beam size for Whisper decoding.')
    parser.add_argument('--whisper_vad_filter', type=lambda x: (str(x).lower() == 'true'), default=True, help='Enable VAD filter for Whisper.')

    # LLM (OllamaLLM)
    parser.add_argument('--ollama_host', type=str, default='http://localhost:11434', help='Ollama service host URL.')
    parser.add_argument('--ollama_vision_model', type=str, default=None, help='Ollama vision model name (e.g., llava, qwen:7b-chat-v1.5-q4_K_M). Set to None if not used.')
    parser.add_argument('--ollama_language_model', type=str, default='qwen:0.5b', help='Ollama language model name (e.g., qwen:0.5b, llama3:8b).')
    parser.add_argument('--sentence_terminators', type=str, default="。？！，,!?;", help="Characters to split LLM stream for TTS.")


    # MuseTalk (MuseReal)
    parser.add_argument('--musetalk_model_dir', type=str, default='./models/musetalk_hf', help='MuseTalk HuggingFace model root directory.')
    parser.add_argument('--musetalk_config_path', type=str, default=None, help='Path to MuseTalk configuration.json. If None, derived from model_dir.')
    parser.add_argument('--musetalk_source_image', type=str, default='./data/avatars/musetalk_default.png', help='Source image for MuseTalk.')

    # System-wide audio sample rate
    parser.add_argument('--audio_sample_rate', type=int, default=16000, help="System target audio sample rate for TTS output and audio processing by BaseReal.")

    opt_global = parser.parse_args()

    # 补全 musetalk_config_path (如果未提供)
    if opt_global.model == 'musetalk' and not opt_global.musetalk_config_path:
        opt_global.musetalk_config_path = os.path.join(opt_global.musetalk_model_dir, 'musetalk/configuration.json')
        logger.info(f"MuseTalk config path defaulted to: {opt_global.musetalk_config_path}")

    # --- 初始化全局模块 ---
    logger.info("Initializing global ASR, LLM, and TTS modules...")
    try:
        # BaseASR 的 __init__ 需要 opt, fps, batch_size, l, r
        # 这些参数对于 WhisperASR 的 transcribe 模式来说不那么重要，但父类构造函数可能需要它们
        # 我们可以在 WhisperASR 的 __init__ 中用默认值调用 super().__init__
        # 或者确保 opt_global 有这些字段
        if not hasattr(opt_global, 'batch_size'): opt_global.batch_size = 1
        if not hasattr(opt_global, 'l'): opt_global.l = 0
        if not hasattr(opt_global, 'r'): opt_global.r = 0

        asr_instance_global = WhisperASR(opt_global)
        llm_instance_global = OllamaLLM(opt_global)
        # TTSEngine 初始化需要 opt
        # BaseReal 内部的 TTS 初始化逻辑可以作为参考
        # TTSEngine(opt, parent_real_instance=None)
        tts_engine_global = TTSEngine(opt_global, None) # parent_real_instance 设为 None，因为它现在是全局引擎
        logger.info("Global ASR, LLM, and TTS modules initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize global modules: {e}", exc_info=True)
        exit(1) # 如果核心模块初始化失败，则退出

    # --- 配置 aiohttp 应用 ---
    app_aiohttp = web.Application(client_max_size=1024**2*100) # 100MB limit for uploads
    app_aiohttp.on_shutdown.append(on_aiohttp_shutdown)

    # 添加路由
    app_aiohttp.router.add_post("/offer", offer_handler)
    app_aiohttp.router.add_post("/process_audio", process_audio_handler) # 新的核心处理端点
    app_aiohttp.router.add_post("/human_text_input", human_text_input_handler) # 用于直接文本输入
    app_aiohttp.router.add_post("/interrupt_talk", interrupt_talk_handler)
    app_aiohttp.router.add_post("/is_speaking", is_speaking_handler)
    
    # 静态文件服务 (web 目录)
    static_files_path = os.path.join(os.path.dirname(__file__), 'web')
    if os.path.exists(static_files_path):
        app_aiohttp.router.add_static('/', path=static_files_path, name='static_root')
        app_aiohttp.router.add_get('/', lambda req: web.HTTPFound('/index.html')) # Redirect / to /index.html or other default
        logger.info(f"Serving static files from: {static_files_path}")
    else:
        logger.warning(f"Static files directory 'web' not found at {static_files_path}")


    # CORS 设置
    cors = aiohttp_cors.setup(app_aiohttp, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*", # 允许所有方法，包括 OPTIONS
        )
    })
    for route in list(app_aiohttp.router.routes()):
        cors.add(route)

    # 启动服务器
    # 旧的 run_server 和 Thread(target=run_server, ...) 逻辑被 web.run_app 替代
    # web.run_app 会处理事件循环
    
    # 旧的启动逻辑:
    # def run_server_sync(runner):
    #     loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(loop)
    #     loop.run_until_complete(runner.setup())
    #     site = web.TCPSite(runner, '0.0.0.0', opt_global.listen_port)
    #     loop.run_until_complete(site.start())
    #     # ... rtcpush logic (omitted for now) ...
    #     loop.run_forever()
    # Thread(target=run_server_sync, args=(web.AppRunner(app_aiohttp),), daemon=True).start()
    # logger.info(f"HTTP server starting on http://0.0.0.0:{opt_global.listen_port}")
    # # Keep main thread alive for the daemon thread
    # try:
    #     while True:
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     logger.info("Keyboard interrupt received, shutting down...")
    #     # Cleanup logic for daemon thread might be needed here if not handled by on_shutdown
    
    logger.info(f"Starting HTTP server on http://0.0.0.0:{opt_global.listen_port}")
    logger.info(f"Recommended client: http://localhost:{opt_global.listen_port}/webrtcapi-asr.html (if it's adapted for POST audio)")

    web.run_app(app_aiohttp, host='0.0.0.0', port=opt_global.listen_port)

    logger.info("Server has shut down.")

# --- BaseReal 需要的修改 (示意，应在 basereal.py 中实现) ---
# class BaseReal:
#     def __init__(self, opt):
#         # ... existing init ...
#         self.aio_audio_track_queue = None
#         self.aio_video_track_queue = None
#         self.event_loop_for_tracks = None # Store the loop

#     def set_event_loop(self, loop):
#         self.event_loop_for_tracks = loop

#     def register_track_queues(self, kind, queue_instance):
#         if kind == "audio":
#             self.aio_audio_track_queue = queue_instance
#             logger.debug(f"Session {self.sessionid}: Audio track queue registered.")
#         elif kind == "video":
#             self.aio_video_track_queue = queue_instance
#             logger.debug(f"Session {self.sessionid}: Video track queue registered.")

#     def start(self):
#         # ... 启动 self._run 和 self.process_frames 的线程 ...
#         # 确保 self.event_loop_for_tracks 在 process_frames 线程中可用或被正确传递
#         pass

#     def stop(self):
#         self.closed = True # 信号给 _run 和 process_frames 线程
#         # ... 等待线程结束 ...
#         logger.info(f"Session {self.sessionid}: BaseReal instance stopped.")


#     # 在 process_frames 中:
#     # async def push_to_aio_queue(self, queue, item):
#     #     if queue:
#     #         await queue.put(item)
#     # ...
#     # if self.aio_video_track_queue and self.event_loop_for_tracks:
#     #    asyncio.run_coroutine_threadsafe(self.push_to_aio_queue(self.aio_video_track_queue,(new_frame,None)), self.event_loop_for_tracks)
#     # else:
#     #    logger.warning(f"Session {self.sessionid}: Video track queue or event loop not set for pushing frame.")
#     # 类似地处理音频帧
#
#     def receive(self, audio_chunk_bytes): # 新增，用于接收TTS的音频
#        if hasattr(self, 'audio_queue') and isinstance(self.audio_queue, mp.Queue): # musereal.py 使用的是 self.audio_queue
#            self.audio_queue.put(audio_chunk_bytes)
#        elif hasattr(self, 'tts_audio_input_queue'): # 或者一个专门的队列
#            self.tts_audio_input_queue.put(audio_chunk_bytes)
#        else:
#            logger.warning(f"Session {self.sessionid}: BaseReal has no audio_queue to receive TTS audio.")

# --- TTSEngine 需要的修改 (示意，应在 ttsreal.py 中实现) ---
# class TTSEngine:
#     async def synthesize_async(self, text: str) -> bytes | None:
#         # 实现异步版本的 synthesize
#         # 例如，如果内部调用是阻塞的，可以使用 asyncio.to_thread
#         # loop = asyncio.get_event_loop()
#         # audio_bytes = await loop.run_in_executor(None, self.synthesize, text)
#         # return audio_bytes
#         # 或者如果 TTS 库本身支持异步：
#         # return await self.tts_lib_async_synthesize(text)
#         # 暂时返回一个同步调用作为占位符，但理想情况下应为异步

#         # 临时的同步转异步包装
#         # 在实际应用中，如果 self.synthesize 是CPU密集型或IO密集型，应该用 run_in_executor
#         try:
#             # return self.synthesize(text) # 如果synthesize非常快，可以直接调用
#             loop = asyncio.get_running_loop() # Python 3.7+
#             audio_bytes = await loop.run_in_executor(None, self.synthesize, text)
#             return audio_bytes
#         except Exception as e:
#             logger.error(f"TTSEngine synthesize_async error: {e}", exc_info=True)
#             return None
```

**关键修改点回顾**：

1.  **全局模块**: `opt_global`, `asr_instance_global`, `llm_instance_global`, `tts_engine_global` 在 `if __name__ == '__main__':` 中初始化。
2.  **`HumanPlayer` 和 Tracks**: `HumanPlayer`, `NerfAudioTrack`, `NerfVideoTrack` 的定义被移入并调整，以适配 `aiohttp` 和 `aiortc` 的异步特性。`BaseReal` 将需要 `register_track_queues` 和 `set_event_loop` 方法，并且其 `process_frames` 内部推送到队列的逻辑需要使用 `asyncio.run_coroutine_threadsafe`。这部分的修改提示已在注释中给出，实际需要修改 `basereal.py`。
3.  **`build_nerfreal_instance`**: 此函数现在负责创建 `MuseReal` (或其他 `BaseReal` 子类) 的实例，并启动它们的内部线程。它不再依赖全局的 `model` 和 `avatar` 变量。
4.  **`offer_handler`**: WebRTC 信令处理函数，现在调用 `build_nerfreal_instance`。
5.  **`process_audio_handler` (新)**: 这是核心的HTTP端点，对应您PPT中的 `handle_message` 下的 `audio_chunk` 逻辑。
    *   接收Base64编码的WAV音频数据和 `sessionid`。
    *   解码音频 -> ASR转写 -> LLM流式处理 -> TTS合成 (分块) -> `real_instance.receive(tts_audio_bytes)`。
    *   TTS 合成调用 `tts_engine_global.synthesize_async` (假设 `TTSEngine` 有此异步方法)。
6.  **参数解析**: 添加了所有新的命令行参数。
7.  **`main` 函数**: 初始化全局模块，设置 `aiohttp` 应用路由，并启动服务器。
8.  **`BaseReal.receive()`**: `BaseReal` 需要一个 `receive(audio_chunk_bytes)` 方法，以便 `app.py` 可以将TTS的输出传递给它。此方法应将音频块放入供 `MuseReal._run()` 使用的内部队列中。
9.  **`TTSEngine.synthesize_async()`**: `TTSEngine` 需要一个异步的合成方法。

**重要提示**：

*   **`basereal.py` 的修改**: 为了使 `HumanPlayer` 和 `aiortc` Tracks 正常工作，`basereal.py` 中的 `BaseReal` 类需要进行配套修改，如上文注释中提到的 `register_track_queues`, `set_event_loop`，以及调整 `process_frames` 中向轨道队列推送数据的方式。还需要添加 `receive()` 和 `stop()` 方法。
*   **`ttsreal.py` 的修改**: `TTSEngine` 需要一个 `synthesize_async` 方法。
*   **前端适配**: 前端 `webrtcapi-asr.html` (或其他客户端) 需要修改：
    *   在WebRTC连接成功并获取 `sessionid` 后。
    *   将录制的音频数据（应为完整的WAV文件内容）进行Base64编码。
    *   通过HTTP POST请求发送到后端的 `/process_audio` 端点，请求体为 JSON，包含 `sessionid` 和 `audio_data_base64`。
*   **错误处理和日志**: 已添加了更多的日志和基本的错误处理。

这次修改非常广泛，因为它试图将一个基于类和明确数据流的设计（如您的PPT所示）整合到一个现有的、不同架构风格（过程式、全局变量、特定WebRTC实现）的代码库中。

我将标记此步骤完成，但请注意，后续可能需要根据测试结果对 `app.py` 以及相关的 `basereal.py` 和 `ttsreal.py` 进行调试和调整。
