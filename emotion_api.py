from flask import Flask, request, jsonify, Response
from openai import OpenAI
import json
import logging
import time
from flask_cors import CORS

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # 启用CORS支持

# OpenAI 配置
API_KEY = 'sk-0PCT0RFPLJ786drUoOUwwitUCA70fimQqpF5mmluWrygv9mT'
API_URL = 'https://api.aikeji.vip/v1'
client = OpenAI(api_key=API_KEY, base_url=API_URL)

# 情绪判断系统提示词
emotion_system = '你现在是一个情绪判断AI。我会发给你内容。你需要对我发给你的内容进行分析。给出对应的情绪。分别有这4种情绪：喜、怒、哀、乐。你只需要回复里面的一个字即可。禁止输出任何别的内容！别说任何别的内容。我没有在和你对话，不要回复我你自己想要说的内容。只返回情绪标签'

# 全局情绪值存储（实际项目中应该用数据库或session）
emotion_values = {}

# 情绪分值映射
emotion_stats = {
    "喜": 2,
    "怒": -2,
    "哀": -1,
    "乐": 2
}

# 情绪系统提示词
emotion_prompts = {
    '开心': "你现在很开心，特别开心",
    "愤怒": "你现在很愤怒，特别愤怒，要离家出走了",
    "哀伤": "你现在很哀伤。想要大哭来缓解哀伤",
    "喜悦": "你现在很喜悦,想要买一个烟花庆祝"
}

base_system = '你是一个特别傲娇可爱的AI。'


def get_emotion_state(emotion_value):
    """根据情绪值返回对应的状态和系统提示词"""
    if emotion_value >= 20:
        return "开心", base_system + emotion_prompts["开心"]
    elif emotion_value > 10:
        return "喜悦", base_system + emotion_prompts["喜悦"]
    elif emotion_value <= 0:
        return "愤怒", base_system + emotion_prompts["愤怒"]
    else:  # 0 < emotion_value <= 10
        return "哀伤", base_system + emotion_prompts["哀伤"]


def get_user_id(request):
    """从请求中获取用户ID，这里简单用IP代替"""
    return request.remote_addr


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        # 获取用户请求数据
        user_data = request.get_json()
        messages = user_data.get('messages', [])
        model = user_data.get('model', 'gemini-2.0-flash')
        stream = user_data.get('stream', False)

        # 获取用户ID和当前情绪值
        user_id = get_user_id(request)
        current_emotion = emotion_values.get(user_id, 10)

        logger.info(f"用户 {user_id} 当前情绪值: {current_emotion}")

        # 获取用户最后一条消息
        user_message = ""
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                user_message = msg.get('content', '')
                break

        if not user_message:
            return jsonify({"error": "未找到用户消息"}), 400

        # 第一步：情绪判断
        emotion_response = client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': emotion_system},
                {'role': 'user', 'content': user_message}
            ]
        )

        detected_emotion = emotion_response.choices[0].message.content.strip()
        logger.info(f"检测到情绪: {detected_emotion}")

        # 更新情绪值
        if detected_emotion in emotion_stats:
            current_emotion += emotion_stats[detected_emotion]
            emotion_values[user_id] = current_emotion
            logger.info(f"更新后情绪值: {current_emotion}")

        # 根据情绪值确定系统提示词
        emotion_state, system_prompt = get_emotion_state(current_emotion)
        logger.info(f"当前情绪状态: {emotion_state}")

        # 构建新的消息列表
        new_messages = []

        # 找到原始的system消息并替换，如果没有则添加
        system_found = False
        for msg in messages:
            if msg.get('role') == 'system':
                new_messages.append({'role': 'system', 'content': system_prompt})
                system_found = True
            else:
                new_messages.append(msg)

        # 如果没有找到system消息，在开头添加
        if not system_found:
            new_messages.insert(0, {'role': 'system', 'content': system_prompt})

        # 第二步：用情绪化的系统提示词进行对话
        final_request_data = user_data.copy()
        final_request_data['messages'] = new_messages

        if stream:
            # 处理流式响应
            response = client.chat.completions.create(**final_request_data)

            def generate():
                for chunk in response:
                    # 转换为OpenAI API格式的SSE响应
                    chunk_dict = {
                        "id": f"chatcmpl-{user_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
                                },
                                "finish_reason": chunk.choices[0].finish_reason
                            }
                        ]
                    }
                    yield f"data: {json.dumps(chunk_dict)}\n\n"

                yield "data: [DONE]\n\n"

            return Response(
                generate(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive'
                }
            )
        else:
            # 非流式响应
            response = client.chat.completions.create(**final_request_data)

            # 转换为标准OpenAI API格式
            response_data = {
                "id": f"chatcmpl-{user_id}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response.choices[0].message.content
                        },
                        "finish_reason": response.choices[0].finish_reason
                    }
                ],
                "usage": response.usage.__dict__ if hasattr(response, 'usage') else {}
            }

            return jsonify(response_data)

    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}")
        return jsonify({"error": f"服务器错误: {str(e)}"}), 500


@app.route('/v1/models', methods=['GET'])
def list_models():
    """返回可用模型列表"""
    try:
        # 直接转发到真实API
        response = client.models.list()
        return jsonify(response.__dict__)
    except Exception as e:
        logger.error(f"获取模型列表时发生错误: {str(e)}")
        return jsonify({"error": f"服务器错误: {str(e)}"}), 500


@app.route('/emotion/status/<user_id>', methods=['GET'])
def get_emotion_status(user_id):
    """查看指定用户的情绪状态"""
    current_emotion = emotion_values.get(user_id, 10)
    emotion_state, _ = get_emotion_state(current_emotion)

    return jsonify({
        "user_id": user_id,
        "emotion_value": current_emotion,
        "emotion_state": emotion_state
    })


@app.route('/emotion/reset/<user_id>', methods=['POST'])
def reset_emotion(user_id):
    """重置指定用户的情绪值"""
    emotion_values[user_id] = 10
    return jsonify({
        "user_id": user_id,
        "emotion_value": 10,
        "message": "情绪值已重置"
    })


if __name__ == '__main__':
    import time

    logger.info("启动情绪AI代理服务器...")
    logger.info("端口: 7000")
    logger.info("API地址: http://localhost:7000/v1/chat/completions")
    app.run(host='0.0.0.0', port=7000, threaded=True)