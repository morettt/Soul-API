from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import json
import logging
import time
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI 配置
API_KEY = 'sk-0PCT0RFPLJ786drUoOUwwitUCA70fimQqpF5mmluWrygv9mT'
API_URL = 'https://api.aikeji.vip/v1'
client = OpenAI(api_key=API_KEY, base_url=API_URL)

# 情绪判断系统提示词
emotion_system = '你现在是一个情绪判断AI。我会发给你内容。你需要对我发给你的内容进行分析。给出对应的情绪。分别有这4种情绪：喜、怒、哀、乐。你只需要回复里面的一个字即可。禁止输出任何别的内容！别说任何别的内容。我没有在和你对话，不要回复我你自己想要说的内容。只返回情绪标签'

# 全局情绪值存储
emotion_values = {}

# 情绪分值映射
emotion_stats = {
    "喜": 2,
    "怒": -2,
    "哀": -1,
    "乐": 2
}

# 情绪系统提示词（只有情绪部分）
emotion_prompts = {
    '开心': "你现在很开心，特别开心",
    "愤怒": "你现在很愤怒，特别愤怒，要离家出走了",
    "哀伤": "你现在很哀伤。想要大哭来缓解哀伤",
    "喜悦": "你现在很喜悦,想要买一个烟花庆祝"
}

def get_emotion_state(emotion_value):
    """根据情绪值返回对应的状态和情绪提示词"""
    if emotion_value >= 20:
        return "开心", emotion_prompts["开心"]
    elif emotion_value > 10:
        return "喜悦", emotion_prompts["喜悦"]
    elif emotion_value <= 0:
        return "愤怒", emotion_prompts["愤怒"]
    else:  # 0 < emotion_value <= 10
        return "哀伤", emotion_prompts["哀伤"]


def get_user_id(request: Request):
    """从请求中获取用户ID，这里简单用IP代替"""
    return request.client.host


# 请求模型
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = 'gemini-2.0-flash'
    stream: bool = False


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, req: Request):
    try:
        # 获取用户ID和当前情绪值
        user_id = get_user_id(req)
        current_emotion = emotion_values.get(user_id, 10)

        logger.info(f"用户 {user_id} 当前情绪值: {current_emotion}")

        # 获取用户最后一条消息
        user_message = ""
        for msg in reversed(request.messages):
            if msg.role == 'user':
                user_message = msg.content
                break

        if not user_message:
            raise HTTPException(status_code=400, detail="未找到用户消息")

        # 第一步：情绪判断
        emotion_response = client.chat.completions.create(
            model=request.model,
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

        # 根据情绪值获取情绪提示词
        emotion_state, emotion_prompt = get_emotion_state(current_emotion)
        logger.info(f"当前情绪状态: {emotion_state}")

        # 构建新的消息列表 - 将情绪提示词添加到客户端的系统提示词中
        new_messages = []
        system_found = False

        for msg in request.messages:
            if msg.role == 'system':
                # 将情绪提示词添加到客户端的系统提示词中
                combined_system = msg.content + " " + emotion_prompt
                new_messages.append({'role': 'system', 'content': combined_system})
                system_found = True
            else:
                new_messages.append({'role': msg.role, 'content': msg.content})

        # 如果没有系统提示词，只添加情绪提示词
        if not system_found:
            new_messages.insert(0, {'role': 'system', 'content': emotion_prompt})

        # 第二步：用组合后的系统提示词进行对话
        if request.stream:
            # 流式响应
            def generate():
                try:
                    response = client.chat.completions.create(
                        model=request.model,
                        messages=new_messages,
                        stream=True
                    )

                    for chunk in response:
                        chunk_dict = {
                            "id": f"chatcmpl-{user_id}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": chunk.choices[0].delta.content if chunk.choices[
                                            0].delta.content else ""
                                    },
                                    "finish_reason": chunk.choices[0].finish_reason
                                }
                            ]
                        }
                        yield f"data: {json.dumps(chunk_dict, ensure_ascii=False)}\n\n"

                    yield "data: [DONE]\n\n"
                except Exception as e:
                    logger.error(f"流式响应错误: {str(e)}")
                    error_chunk = {
                        "error": {"message": str(e), "type": "server_error"}
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                }
            )
        else:
            # 非流式响应
            response = client.chat.completions.create(
                model=request.model,
                messages=new_messages
            )

            response_data = {
                "id": f"chatcmpl-{user_id}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
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
                "usage": response.usage.dict() if hasattr(response, 'usage') and response.usage else {}
            }

            return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


@app.get("/v1/models")
async def list_models():
    """返回可用模型列表"""
    try:
        response = client.models.list()
        return JSONResponse(content=response.dict())
    except Exception as e:
        logger.error(f"获取模型列表时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


@app.get("/emotion/status/{user_id}")
async def get_emotion_status(user_id: str):
    """查看指定用户的情绪状态"""
    current_emotion = emotion_values.get(user_id, 10)
    emotion_state, _ = get_emotion_state(current_emotion)

    return {
        "user_id": user_id,
        "emotion_value": current_emotion,
        "emotion_state": emotion_state
    }


@app.post("/emotion/reset/{user_id}")
async def reset_emotion(user_id: str):
    """重置指定用户的情绪值"""
    emotion_values[user_id] = 10
    return {
        "user_id": user_id,
        "emotion_value": 10,
        "message": "情绪值已重置"
    }


if __name__ == '__main__':
    import uvicorn

    logger.info("启动情绪AI代理服务器...")
    logger.info("端口: 7000")
    logger.info("API地址: http://localhost:7000/v1/chat/completions")
    uvicorn.run(app, host='0.0.0.0', port=7000)
