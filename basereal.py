

import math
import torch
import numpy as np
import subprocess
import os
import time
import cv2
import glob
import resampy # Ensure resampy is in requirements.txt
import queue # Standard queue
from queue import Queue as ThreadQueue
from threading import Thread, Event
from io import BytesIO
import soundfile as sf
import asyncio
from av import AudioFrame, VideoFrame # av for WebRTC frame types
from logger import logger
from tqdm import tqdm
import torch.multiprocessing as mp # Added for mp.Event, though threading.Event is used later
from typing import Any # For type hinting

# Removed TTS specific imports from here, as TTS is now external

def read_imgs(img_list):
    frames = []
    logger.info('Reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        if frame is None:
            logger.warning(f"Warning: Failed to read image {img_path}. Skipping.")
            continue
        frames.append(frame)
    return frames

# play_audio function is removed as audio playback is handled by WebRTC client or other means.

class BaseReal:
    def __init__(self, opt):
        self.opt = opt
        self.system_audio_sample_rate = getattr(opt, 'audio_sample_rate', 16000)
        self.fps = getattr(opt, 'fps', 25)
        self.audio_chunk_samples = self.system_audio_sample_rate // self.fps
        
        self.sessionid = getattr(opt, 'sessionid', 'default_session')
        self.closed = Event()

        self.speaking = False
        self.recording = False
        self._record_video_pipe = None
        self._record_audio_pipe = None

        self.width = getattr(self.opt, 'W', 0)
        self.height = getattr(self.opt, 'H', 0)
        if self.width == 0 or self.height == 0:
            logger.info(f"Session {self.sessionid}: Initial video dimensions W/H not specified in opt, will be set on first frame.")

        self.audio_input_queue = ThreadQueue(maxsize=100)
        self.aio_audio_track_queue = None
        self.aio_video_track_queue = None
        self.event_loop_for_tracks = None
        self.res_frame_queue = ThreadQueue(maxsize=getattr(self.opt, 'batch_size', 2) * 2)

        self.send_keepalive_frames = getattr(opt, 'send_keepalive_frames', True)
        self.keepalive_timeout_seconds = getattr(opt, 'keepalive_timeout_seconds', 2.0)
        self.keepalive_frame_interval = getattr(opt, 'keepalive_frame_interval', 1.0)

        self.curr_state=0
        self.custom_img_cycle = {}
        self.custom_audio_cycle = {}
        self.custom_audio_index = {}
        self.custom_index = {}
        self.custom_opt_map = {}
        if hasattr(opt, 'customopt') and opt.customopt:
            self.__loadcustom()

        self._process_frames_thread = None
        self._run_thread = None
        logger.info(f"BaseReal (Session: {self.sessionid}) initialized. System SR: {self.system_audio_sample_rate}Hz, FPS: {self.fps}")

    def receive(self, audio_chunk_bytes: bytes):
        if not self.closed.is_set():
            try:
                self.audio_input_queue.put(audio_chunk_bytes, timeout=0.1)
                logger.debug(f"Session {self.sessionid}: Received {len(audio_chunk_bytes)} bytes into audio_input_queue.")
            except queue.Full:
                logger.warning(f"Session {self.sessionid}: Audio input queue full. Discarding audio chunk.")
        else:
            logger.warning(f"Session {self.sessionid}: Attempted to receive audio on a closed instance.")

    def register_track_queues(self, kind: str, track_queue: asyncio.Queue):
        if kind == "audio": self.aio_audio_track_queue = track_queue
        elif kind == "video": self.aio_video_track_queue = track_queue
        logger.info(f"Session {self.sessionid}: Asyncio {kind} track queue registered.")

    def set_event_loop_for_tracks(self, loop: asyncio.AbstractEventLoop):
        self.event_loop_for_tracks = loop
        logger.info(f"Session {self.sessionid}: Event loop for tracks registered.")

    def start(self):
        if self._run_thread or self._process_frames_thread:
            logger.warning(f"Session {self.sessionid}: Start called but threads may already be running.")
            return
        self.closed.clear()
        if hasattr(self, '_run') and callable(self._run):
            self._run_thread = Thread(target=self._thread_wrapper, args=(self._run, "RunLoop"), name=f"BaseReal_Run_{self.sessionid}")
            self._run_thread.daemon = True
            self._run_thread.start()
        else:
            logger.error(f"Session {self.sessionid}: _run method not implemented or not callable. Avatar driver will not start.")

        self._process_frames_thread = Thread(target=self._thread_wrapper, args=(self.process_frames, "ProcessFramesLoop"), name=f"BaseReal_ProcessFrames_{self.sessionid}")
        self._process_frames_thread.daemon = True
        self._process_frames_thread.start()
        logger.info(f"Session {self.sessionid}: BaseReal threads started.")

    def _thread_wrapper(self, target_func, loop_name: str):
        logger.info(f"Thread {loop_name} (Session {self.sessionid}) starting.")
        try:
            target_func()
        except Exception as e:
            logger.error(f"CRITICAL ERROR in thread {loop_name} (Session {self.sessionid}): {e}", exc_info=True)
        finally:
            logger.info(f"Thread {loop_name} (Session {self.sessionid}) finished.")

    def stop(self):
        logger.info(f"Session {self.sessionid}: Stop called. Signaling threads to close.")
        self.closed.set()
        thread_timeout = 2.0
        if self._run_thread and self._run_thread.is_alive():
            logger.debug(f"Session {self.sessionid}: Waiting for _run thread to join...")
            self._run_thread.join(timeout=thread_timeout)
            if self._run_thread.is_alive(): logger.warning(f"Session {self.sessionid}: _run thread did not join in time.")
        self._run_thread = None
        if self._process_frames_thread and self._process_frames_thread.is_alive():
            logger.debug(f"Session {self.sessionid}: Waiting for process_frames thread to join...")
            self._process_frames_thread.join(timeout=thread_timeout)
            if self._process_frames_thread.is_alive(): logger.warning(f"Session {self.sessionid}: process_frames thread did not join in time.")
        self._process_frames_thread = None
        try:
            while not self.audio_input_queue.empty(): self.audio_input_queue.get_nowait()
            while not self.res_frame_queue.empty(): self.res_frame_queue.get_nowait()
        except queue.Empty: pass
        logger.info(f"Session {self.sessionid}: BaseReal instance stopped and queues cleared.")

    def is_speaking(self) -> bool:
        return self.speaking

    def flush_talk(self):
        logger.info(f"Session {self.sessionid}: Flushing talk. Clearing audio_input_queue.")
        try:
            while not self.audio_input_queue.empty(): self.audio_input_queue.get_nowait()
        except queue.Empty: pass

    def get_audio_chunk(self, timeout=0.02) -> bytes | None:
        if self.closed.is_set(): return None
        try:
            return self.audio_input_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

    def _push_to_aio_queue(self, q: asyncio.Queue, item: Any, item_type_for_log: str = "item"):
        if q and self.event_loop_for_tracks and self.event_loop_for_tracks.is_running() and not self.closed.is_set():
            future = asyncio.run_coroutine_threadsafe(q.put(item), self.event_loop_for_tracks)
            try:
                future.result(timeout=0.1)
            except asyncio.TimeoutError:
                logger.warning(f"Session {self.sessionid}: Timeout pushing {item_type_for_log} to asyncio queue (size: {q.qsize() if q else 'N/A'}).")
            except RuntimeError as e:
                 logger.error(f"Session {self.sessionid}: Runtime error pushing {item_type_for_log} to asyncio queue: {e}")
            except Exception as e:
                logger.error(f"Session {self.sessionid}: Error pushing {item_type_for_log} to asyncio queue: {e}", exc_info=True)
        elif self.closed.is_set():
            logger.debug(f"Session {self.sessionid}: Instance closed, not pushing {item_type_for_log} to aio queue.")
        elif not (q and self.event_loop_for_tracks and self.event_loop_for_tracks.is_running()):
            logger.warning(f"Session {self.sessionid}: Cannot push {item_type_for_log} to aio_queue: Queue (set: {q is not None}), event loop (set: {self.event_loop_for_tracks is not None}, running: {self.event_loop_for_tracks.is_running() if self.event_loop_for_tracks else 'N/A'}).")

    def push_frame(self, video_frame_bgr: np.ndarray, audio_frame_np: np.ndarray = None, eventpoint=None):
        if self.closed.is_set(): return

        if video_frame_bgr is not None and self.aio_video_track_queue:
            try:
                if self.width == 0 or self.height == 0:
                    self.height, self.width = video_frame_bgr.shape[:2]
                    logger.info(f"Session {self.sessionid}: Video dimensions set from first frame: {self.width}x{self.height}")

                current_h, current_w = video_frame_bgr.shape[:2]
                if current_w != self.width or current_h != self.height:
                    logger.warning(f"Session {self.sessionid}: Video frame ({current_w}x{current_h}) differs from target ({self.width}x{self.height}). Resizing.")
                    video_frame_bgr = cv2.resize(video_frame_bgr, (self.width, self.height))

                new_video_frame = VideoFrame.from_ndarray(video_frame_bgr, format="bgr24")
                self._push_to_aio_queue(self.aio_video_track_queue, (new_video_frame, eventpoint), "video_frame")
            except Exception as e: logger.error(f"Session {self.sessionid}: Failed to create or push video frame: {e}", exc_info=True)

        if audio_frame_np is not None and self.aio_audio_track_queue:
            try:
                audio_frame_int16 = audio_frame_np
                if audio_frame_np.dtype == np.float32:
                    audio_frame_int16 = (audio_frame_np * 32767).astype(np.int16)
                elif audio_frame_np.dtype != np.int16:
                    logger.warning(f"Session {self.sessionid}: Unsupported audio frame dtype {audio_frame_np.dtype} for push_frame.")
                    return

                new_audio_frame = AudioFrame(format='s16', layout='mono', samples=audio_frame_int16.shape[0])
                new_audio_frame.planes[0].update(audio_frame_int16.tobytes())
                new_audio_frame.sample_rate = self.system_audio_sample_rate
                self._push_to_aio_queue(self.aio_audio_track_queue, (new_audio_frame, eventpoint), "audio_frame")
            except Exception as e: logger.error(f"Session {self.sessionid}: Failed to create or push audio frame: {e}", exc_info=True)

    # --- Deprecated / Replaced methods from original BaseReal ---
    def put_msg_txt(self,msg,eventpoint=None):
        logger.warning("put_msg_txt is deprecated in BaseReal. Text input is handled by app.py's pipeline.")
    
    def put_audio_frame(self,audio_chunk,eventpoint=None):
        logger.warning("put_audio_frame is deprecated in BaseReal. ASR input is handled by app.py.")

    def put_audio_file(self,filebyte): 
        logger.warning("put_audio_file is deprecated in BaseReal. ASR input is handled by app.py.")
    
    def __create_bytes_stream(self,byte_stream):
        logger.warning("__create_bytes_stream is deprecated. Audio processing handled differently now.")
        return np.array([], dtype=np.float32) # Return empty to avoid breaking old calls if any

    # --- Methods for custom animations/audio and recording (kept from original, with improvements) ---
    def __loadcustom(self):
        logger.info(f"Session {self.sessionid}: Loading custom animations/audio from opt.customopt: {self.opt.customopt}")
        for item_spec in self.opt.customopt:
            audiotype = item_spec.get('audiotype')
            imgpath = item_spec.get('imgpath')
            audiopath = item_spec.get('audiopath')
            if not all([audiotype, imgpath, audiopath]):
                logger.warning(f"Skipping invalid customopt item (missing fields): {item_spec}")
                continue
            logger.info(f"Loading custom for audiotype '{audiotype}': img='{imgpath}', audio='{audiopath}'")
            try:
                abs_imgpath = os.path.abspath(imgpath)
                abs_audiopath = os.path.abspath(audiopath)
                if not os.path.isdir(abs_imgpath):
                    logger.warning(f"Image path '{abs_imgpath}' for audiotype '{audiotype}' is not a directory.")
                    continue
                if not os.path.isfile(abs_audiopath):
                    logger.warning(f"Audio path '{abs_audiopath}' for audiotype '{audiotype}' not found.")
                    continue

                input_img_list = sorted(glob.glob(os.path.join(abs_imgpath, '*.[jpJP][pnPN]*[gG]')),
                                        key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                if not input_img_list:
                    logger.warning(f"No images found in '{abs_imgpath}' for audiotype '{audiotype}'. Pattern: *.[jpJP][pnPN]*[gG]")
                    continue
                self.custom_img_cycle[audiotype] = read_imgs(input_img_list)
                if not self.custom_img_cycle[audiotype]:
                     logger.warning(f"Failed to read any images from '{abs_imgpath}' for audiotype '{audiotype}'.")
                     del self.custom_img_cycle[audiotype]
                     continue

                custom_audio_data, sr_in = sf.read(abs_audiopath, dtype='float32')
                if sr_in != self.system_audio_sample_rate:
                    logger.warning(f"Custom audio '{abs_audiopath}' SR ({sr_in}Hz) != system SR ({self.system_audio_sample_rate}Hz). Resampling...")
                    custom_audio_data = resampy.resample(custom_audio_data, sr_orig=sr_in, sr_new=self.system_audio_sample_rate)
                self.custom_audio_cycle[audiotype] = custom_audio_data
                self.custom_audio_index[audiotype] = 0
                self.custom_index[audiotype] = 0
                self.custom_opt_map[audiotype] = item_spec
            except Exception as e:
                logger.error(f"Failed to load custom asset for audiotype '{audiotype}': {e}", exc_info=True)

    def init_customindex(self):
        self.curr_state=0
        for key_ci in self.custom_audio_index: self.custom_audio_index[key_ci]=0
        for key_c in self.custom_index: self.custom_index[key_c]=0
        logger.info(f"Session {self.sessionid}: Custom indices re-initialized.")

    def notify(self,eventpoint):
        logger.info(f"Session {self.sessionid}: Eventpoint '{eventpoint}' notified.")

    def start_recording(self):
        if self.recording:
            logger.info(f"Session {self.sessionid}: Recording already in progress.")
            return
        if self.width == 0 or self.height == 0:
            logger.error(f"Session {self.sessionid}: Cannot start recording, video dimensions (W:{self.width},H:{self.height}) not set or invalid.")
            return

        os.makedirs("data", exist_ok=True)
        temp_video_fn = f"temp_video_{self.sessionid}.mp4"
        temp_audio_fn = f"temp_audio_{self.sessionid}.aac"

        video_cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-an', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f"{self.width}x{self.height}", '-r', str(self.fps), '-i', '-', '-pix_fmt', 'yuv420p', '-vcodec', 'libx264', '-preset', 'ultrafast', temp_video_fn]
        self._record_video_pipe = subprocess.Popen(video_cmd, stdin=subprocess.PIPE)

        audio_cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-vn', '-f', 's16le', '-ac', '1', '-ar', str(self.system_audio_sample_rate), '-i', '-', '-acodec', 'aac', '-b:a', '64k', temp_audio_fn]
        self._record_audio_pipe = subprocess.Popen(audio_cmd, stdin=subprocess.PIPE)

        self.recording = True
        logger.info(f"Session {self.sessionid}: Started recording. Video: {self.width}x{self.height}@{self.fps}fps, Audio: {self.system_audio_sample_rate}Hz mono.")

    def record_video_data(self,image_bgr: np.ndarray):
        if self.recording and self._record_video_pipe and self._record_video_pipe.stdin and not self._record_video_pipe.stdin.closed:
            try: self._record_video_pipe.stdin.write(image_bgr.tobytes())
            except BrokenPipeError:
                logger.error(f"Session {self.sessionid}: Broken pipe for video recording FFmpeg process. Stopping recording.");
                self._handle_ffmpeg_pipe_error(self._record_video_pipe)
                self.stop_recording()
            except Exception as e: logger.error(f"Session {self.sessionid}: Error writing video data to FFmpeg: {e}")

    def record_audio_data(self,frame_int16: np.ndarray):
        if self.recording and self._record_audio_pipe and self._record_audio_pipe.stdin and not self._record_audio_pipe.stdin.closed:
            try: self._record_audio_pipe.stdin.write(frame_int16.tobytes())
            except BrokenPipeError:
                logger.error(f"Session {self.sessionid}: Broken pipe for audio recording FFmpeg process. Stopping recording.");
                self._handle_ffmpeg_pipe_error(self._record_audio_pipe)
                self.stop_recording()
            except Exception as e: logger.error(f"Session {self.sessionid}: Error writing audio data to FFmpeg: {e}")
    
    def _handle_ffmpeg_pipe_error(self, ffmpeg_pipe: subprocess.Popen):
        if ffmpeg_pipe:
            return_code = ffmpeg_pipe.poll()
            if return_code is not None: logger.error(f"FFmpeg process terminated unexpectedly with code {return_code}.")

    def stop_recording(self):
        if not self.recording: return
        self.recording = False 
        logger.info(f"Session {self.sessionid}: Stopping recording...")
        ffmpeg_timeout = 5

        if self._record_video_pipe:
            if self._record_video_pipe.stdin and not self._record_video_pipe.stdin.closed: self._record_video_pipe.stdin.close()
            try: self._record_video_pipe.wait(timeout=ffmpeg_timeout)
            except subprocess.TimeoutExpired: logger.warning(f"Session {self.sessionid}: Timeout waiting for video FFmpeg to exit.")
            self._record_video_pipe = None
        if self._record_audio_pipe:
            if self._record_audio_pipe.stdin and not self._record_audio_pipe.stdin.closed: self._record_audio_pipe.stdin.close()
            try: self._record_audio_pipe.wait(timeout=ffmpeg_timeout)
            except subprocess.TimeoutExpired: logger.warning(f"Session {self.sessionid}: Timeout waiting for audio FFmpeg to exit.")
            self._record_audio_pipe = None

        output_filename = f"data/record_{self.sessionid}_{int(time.time())}.mp4"
        temp_audio_file = f"temp_audio_{self.sessionid}.aac"
        temp_video_file = f"temp_video_{self.sessionid}.mp4"

        if not (os.path.exists(temp_audio_file) and os.path.getsize(temp_audio_file) > 0 and \
                os.path.exists(temp_video_file) and os.path.getsize(temp_video_file) > 0) :
            logger.error(f"Session {self.sessionid}: Temporary recording files are missing, empty, or inaccessible. Cannot combine.")
            logger.error(f"  Audio: '{temp_audio_file}' (Exists: {os.path.exists(temp_audio_file)}, Size: {os.path.getsize(temp_audio_file) if os.path.exists(temp_audio_file) else 'N/A'})")
            logger.error(f"  Video: '{temp_video_file}' (Exists: {os.path.exists(temp_video_file)}, Size: {os.path.getsize(temp_video_file) if os.path.exists(temp_video_file) else 'N/A'})")
        else:
            combine_cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', temp_audio_file, '-i', temp_video_file, '-c:v', 'copy', '-c:a', 'copy', output_filename]
            logger.info(f"Session {self.sessionid}: Combining recorded streams into '{output_filename}'...")
            try:
                result = subprocess.run(combine_cmd, check=True, capture_output=True, text=True)
                logger.info(f"Session {self.sessionid}: Recording saved to '{output_filename}'. FFmpeg output: {result.stdout or result.stderr}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Session {self.sessionid}: Failed to combine audio/video. Command: '{' '.join(e.cmd)}'. Error: {e.stderr}")
            except FileNotFoundError:
                logger.error(f"Session {self.sessionid}: FFmpeg command not found. Ensure FFmpeg is installed and in PATH.")

        for temp_file in [temp_audio_file, temp_video_file]:
            if os.path.exists(temp_file):
                try: os.remove(temp_file)
                except Exception as e_rm: logger.warning(f"Session {self.sessionid}: Could not remove temp file {temp_file}: {e_rm}")
        logger.info(f"Session {self.sessionid}: Recording process finished and temp files cleaned up.")

    def mirror_index(self,size, index): # Kept from original if custom animations use it
        turn = index // size
        res = index % size
        if turn % 2 == 0: return res
        else: return size - res - 1
    
    def get_audio_stream(self,audiotype): # Kept from original for custom animations
        # This method provides audio for custom, non-AI driven states.
        # It should return audio chunks matching self.audio_chunk_samples at self.system_audio_sample_rate
        if audiotype not in self.custom_audio_cycle:
            logger.warning(f"Session {self.sessionid}: Custom audiotype '{audiotype}' not found for get_audio_stream.")
            return np.zeros(self.audio_chunk_samples, dtype=np.float32) # Return silence

        audio_data_full = self.custom_audio_cycle[audiotype]
        current_sample_idx = self.custom_audio_index[audiotype]

        if current_sample_idx >= len(audio_data_full):
            # Audio for this custom state has finished.
            # Behavior depends on requirements: loop, switch to silence, or signal end.
            # Original logic: self.curr_state = 1 (silence/default idle)
            logger.info(f"Session {self.sessionid}: Custom audio for audiotype '{audiotype}' ended. Switching state or providing silence.")
            self.curr_state = 1 # Example: switch to a defined silent/idle state
            return np.zeros(self.audio_chunk_samples, dtype=np.float32)

        end_sample_idx = current_sample_idx + self.audio_chunk_samples
        audio_chunk_np = audio_data_full[current_sample_idx:end_sample_idx]

        # Pad if chunk is too short (e.g., end of audio file)
        if len(audio_chunk_np) < self.audio_chunk_samples:
            padding = np.zeros(self.audio_chunk_samples - len(audio_chunk_np), dtype=np.float32)
            audio_chunk_np = np.concatenate((audio_chunk_np, padding))

        self.custom_audio_index[audiotype] = end_sample_idx # Update index for next call
        return audio_chunk_np # Return float32 numpy array
    
    def set_custom_state(self,audiotype, reinit=True): # Kept from original
        logger.info(f"Session {self.sessionid}: Setting custom state to audiotype '{audiotype}', reinit: {reinit}")
        if audiotype not in self.custom_audio_index: # Check against custom_audio_index or custom_opt_map
            logger.warning(f"Session {self.sessionid}: Attempted to set invalid custom audiotype '{audiotype}'.")
            return
        self.curr_state = audiotype
        if reinit:
            self.custom_audio_index[audiotype] = 0
            self.custom_index[audiotype] = 0 # Also reset image index for this state
            logger.info(f"Session {self.sessionid}: Custom state '{audiotype}' indices reset.")

    def _run(self):
        logger.error(f"Session {self.sessionid}: _run() called on BaseReal itself. Subclasses must implement this.")
        raise NotImplementedError("_run must be implemented by BaseReal subclasses.")