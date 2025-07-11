# ollama_llm.py
# 对接本地 Ollama 服务的 LLM 模块

import ollama
import base64
import os
from logger import logger # 使用项目统一的logger

class OllamaLLM:
    """
    封装对Ollama的调用，支持视觉和语言模型
    """
    def __init__(self, opt):
        """
        初始化Ollama配置
        """
        self.opt = opt
        self.ollama_host = getattr(self.opt, 'ollama_host', 'http://localhost:11434')
        self.vision_model = getattr(self.opt, 'ollama_vision_model', None) # 例如 'llava' 或 'qwen:7b-chat-v1.5-q4_K_M'
        self.language_model = getattr(self.opt, 'ollama_language_model', 'qwen:0.5b') # 默认一个小型号

        logger.info(f"正在初始化 Ollama LLM 客户端，目标主机: {self.ollama_host}")
        try:
            self.client = ollama.Client(host=self.ollama_host)
            # 尝试列出本地模型以验证连接和模型可用性
            self.client.list()
            logger.info(f"Ollama LLM 初始化完毕。")
            logger.info(f"  视觉模型: {self.vision_model if self.vision_model else '未配置'}")
            logger.info(f"  语言模型: {self.language_model}")
        except Exception as e:
            logger.error(f"连接 Ollama 服务 ({self.ollama_host}) 失败或列出模型失败: {e}")
            logger.error("请确保 Ollama 服务正在运行，并且可以通过指定的地址访问。")
            # 可以选择在此处引发异常，或者允许程序继续运行但LLM功能将不可用
            self.client = None # 标记客户端不可用

    def get_visual_description(self, image_path: str, prompt: str) -> str:
        """
        调用视觉模型获取图片描述。
        """
        if not self.client:
            logger.error("OllamaLLM: 客户端未初始化，无法进行视觉描述。")
            return "视觉模型服务不可用。"

        if not self.vision_model:
            logger.warning("OllamaLLM: 未配置视觉模型 (ollama_vision_model)，无法进行视觉描述。")
            return "未配置视觉模型。"

        if not os.path.exists(image_path):
            logger.error(f"OllamaLLM: 图片文件不存在: {image_path}")
            return "指定的图片文件不存在。"

        logger.info(f"OllamaLLM: 正在用视觉模型 '{self.vision_model}' 分析图片 '{image_path}'，提示: '{prompt}'")
        try:
            # ollama库的新版本可能直接支持 image_path 参数，但为了兼容性和明确性，我们使用base64编码
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

            response = self.client.chat(
                model=self.vision_model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [encoded_image], # 部分模型可能期望 images: [path_to_image]
                                               # 如果ollama python库支持，可以直接传路径列表
                                               # 或者如果模型支持，可以传 base64 编码的图片
                }]
            )
            description = response['message']['content']
            logger.info(f"OllamaLLM: 视觉分析结果 - {description}")
            return description
        except Exception as e:
            logger.error(f"OllamaLLM: 视觉模型 ('{self.vision_model}') 调用失败: {e}")
            return "抱歉，我无法分析这张图片。"

    def chat_stream(self, prompt: str, history=None):
        """
        以流式方式调用语言模型生成回复。
        'history' 是一个可选的消息列表，例如:
        [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮你的吗？"}
        ]
        """
        if not self.client:
            logger.error("OllamaLLM: 客户端未初始化，无法进行聊天。")
            yield "LLM 服务不可用。"
            return

        if not self.language_model:
            logger.error("OllamaLLM: 未配置语言模型 (ollama_language_model)，无法进行聊天。")
            yield "语言模型未配置。"
            return

        logger.info(f"OllamaLLM: 正在用语言模型 '{self.language_model}' 处理提示: '{prompt}'")

        messages = []
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        try:
            # stream=True 是关键
            response_stream = self.client.chat(
                model=self.language_model,
                messages=messages,
                stream=True
            )

            full_response_for_log = []
            for chunk in response_stream:
                if chunk['done']: # 检查流是否结束
                    # 'total_duration', 'load_duration', 'prompt_eval_count',
                    # 'prompt_eval_duration', 'eval_count', 'eval_duration'
                    # 可以在 chunk 中找到，用于日志或统计
                    logger.debug(f"OllamaLLM: 流结束. Total duration: {chunk.get('total_duration', 'N/A')}")

                content_piece = chunk['message']['content']
                full_response_for_log.append(content_piece)
                yield content_piece

            logger.info(f"OllamaLLM: 语言模型流式回复完成: '{''.join(full_response_for_log)}'")

        except ConnectionRefusedError:
            logger.error(f"OllamaLLM: 连接 Ollama 服务 ({self.ollama_host}) 被拒绝。请确保服务正在运行。")
            yield "抱歉，连接大模型服务失败。"
        except ollama.ResponseError as e:
            logger.error(f"OllamaLLM: Ollama API 错误 (模型: '{self.language_model}'): {e.status_code} - {e.error}")
            if "model not found" in e.error.lower():
                logger.error(f"  看起来模型 '{self.language_model}' 在 Ollama 中不存在。请运行 'ollama pull {self.language_model}'。")
                yield f"抱歉，语言模型 '{self.language_model}' 未找到。"
            else:
                yield "抱歉，语言模型调用时发生API错误。"
        except Exception as e:
            logger.error(f"OllamaLLM: 语言模型 ('{self.language_model}') 调用失败: {e}")
            yield "抱歉，我的大脑出了一点小问题。"

# 可选：添加一个简单的测试函数
if __name__ == '__main__':
    class MockOpt:
        def __init__(self):
            self.ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
            # 为了测试，请确保Ollama服务中存在这些模型，或者修改为可用的模型
            self.ollama_vision_model = os.getenv('OLLAMA_VISION_MODEL','qwen:0.5b-chat-v1.5-q2_K') # 'llava:7b-v1.6-mistral-q4_0' # 替换为你ollama中已有的视觉模型
            self.ollama_language_model = os.getenv('OLLAMA_LANGUAGE_MODEL', 'qwen:0.5b') # 'llama3:8b' # 替换为你ollama中已有的语言模型
            # 如果没有视觉模型，可以设置为 None
            # self.ollama_vision_model = None

    print("测试 OllamaLLM...")
    mock_opt = MockOpt()
    llm_instance = OllamaLLM(mock_opt)

    if not llm_instance.client:
        print("OllamaLLM 客户端初始化失败，测试中止。请检查 Ollama 服务状态和网络连接。")
    else:
        # 测试语言模型
        print(f"\n--- 测试语言模型 ({llm_instance.language_model}) ---")
        if llm_instance.language_model:
            test_prompt = "你好，给我讲一个关于程序员的短笑话。"
            print(f"发送提示: {test_prompt}")
            full_response = []
            try:
                for piece in llm_instance.chat_stream(test_prompt):
                    print(piece, end='', flush=True)
                    full_response.append(piece)
                print("\n语言模型回复接收完毕。")
                if not full_response:
                    print("警告: 未收到任何回复内容。")
            except Exception as e:
                print(f"\n语言模型测试中发生错误: {e}")
        else:
            print("未配置语言模型，跳过测试。")

        # 测试视觉模型
        print(f"\n--- 测试视觉模型 ({llm_instance.vision_model}) ---")
        if llm_instance.vision_model:
            # 创建一个临时的虚拟图片文件用于测试
            temp_image_path = "temp_test_image.png"
            try:
                from PIL import Image, ImageDraw
                img = Image.new('RGB', (100, 100), color = 'red')
                draw = ImageDraw.Draw(img)
                draw.text((10,10), "Test", fill=(0,0,0))
                img.save(temp_image_path)
                print(f"创建临时测试图片: {temp_image_path}")

                vision_prompt = "这张图片里有什么内容?"
                print(f"发送视觉提示: {vision_prompt} for image {temp_image_path}")
                description = llm_instance.get_visual_description(temp_image_path, vision_prompt)
                print(f"视觉模型回复: {description}")

            except ImportError:
                print("Pillow 未安装，无法创建测试图片。跳过视觉模型文件测试。")
                print("你可以手动创建一个图片，并在这里提供路径来测试。")
            except Exception as e:
                print(f"视觉模型测试中发生错误: {e}")
            finally:
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
                    print(f"删除临时测试图片: {temp_image_path}")
        else:
            print("未配置视觉模型或视觉模型路径无效，跳过测试。")

    print("\nOllamaLLM 测试结束。")
