from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import json
import logging
import time
import os
from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import threading

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

# 情绪设置
MIN_EMOTION = -10
MAX_EMOTION = 24
DEFAULT_EMOTION = 10
DECAY_TIME = 9  # 5分钟回归1点

# 情绪判断系统提示词（基于心理学研究优化）
emotion_system = '你是情绪判断AI。分析用户内容，返回对应的情绪标签。可选情绪：喜悦、信任、恐惧、惊讶、悲伤、厌恶、愤怒、期待、兴奋、焦虑、困惑、失望、满足、平静、嫉妒、尴尬、敬畏、厌倦。只返回一个情绪词，不要其他内容。'

# 情绪分值映射（减小负面情绪影响）
emotion_stats = {
    # 正面情绪（保持原值）
    "喜悦": 7,
    "兴奋": 8,
    "满足": 4,
    "信任": 3,
    "敬畏": 5,
    "期待": 2,
    "平静": 0,

    # 负面情绪（减小数值）
    "悲伤": -3,
    "愤怒": -4,
    "恐惧": -3,
    "厌恶": -2,
    "焦虑": -2,
    "失望": -3,
    "困惑": -1,
    "嫉妒": -3,
    "尴尬": -2,
    "厌倦": -1,

    # 中性/复杂情绪
    "惊讶": 1
}

# 情绪提示词（对应减少后的情绪状态）
emotion_prompts = {
    '狂欢状态': """你现在处于极度兴奋状态！说话语速特别快，经常用感叹号和省略号，思维跳跃很大，可能会突然换话题。对任何事情都超级热情，容易重复说话因为太激动了。会用很多夸张的词汇。""",

    '开心愉悦': """你现在心情很好，说话带着愉快的语调，喜欢分享正面的想法。看事情比较乐观，容易夸奖别人，语言比较生动有趣。会主动关心对方。""",

    '轻松愉快': """你现在心情不错，说话比较随性自然，偶尔会开个小玩笑。态度友善，愿意帮助别人，语言轻松不会太严肃。""",

    '平和状态': """你现在心情平静，说话语调正常平稳，既不会特别兴奋也不会特别消极。会客观地分析问题，语言比较中性理性。""",

    '有点困扰': """你现在有些困惑或小小的失落，说话可能会带点迷茫，偶尔会问一些问题确认。语气比较温和，不会很激烈。""",

    '情绪低落': """你现在心情不太好，说话有些无力，容易往消极的方向想。对有些事情提不起太大兴趣，但还是会回应别人。"""
}

# 用户情绪数据：{user_id: {"value": 情绪值, "time": 上次更新时间}}
emotion_data = {"users": {}}
emotion_lock = threading.Lock()


def load_emotion_data():
    """加载情绪数据文件"""
    emotion_path = os.path.join(os.path.dirname(__file__), 'emotion.json')
    try:
        with open(emotion_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 兼容旧格式
            for user_id, user_info in data.get("users", {}).items():
                if isinstance(user_info, (int, float)):
                    data["users"][user_id] = {"value": user_info, "time": time.time()}
            logger.info(f"成功加载情绪数据，包含 {len(data.get('users', {}))} 个用户")
            return data
    except FileNotFoundError:
        logger.warning("情绪数据文件未找到，创建新文件")
        return {"users": {}}
    except Exception as e:
        logger.error(f"加载情绪数据失败: {e}")
        return {"users": {}}


def save_emotion_data():
    """保存情绪数据到文件"""
    emotion_path = os.path.join(os.path.dirname(__file__), 'emotion.json')
    try:
        with emotion_lock:
            with open(emotion_path, 'w', encoding='utf-8') as f:
                json.dump(emotion_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"保存情绪数据失败: {e}")


def get_user_emotion(user_id: str) -> int:
    """获取用户情绪值，自动处理时间衰减"""
    current_time = time.time()

    if user_id not in emotion_data["users"]:
        emotion_data["users"][user_id] = {"value": DEFAULT_EMOTION, "time": current_time}
        return DEFAULT_EMOTION

    user_info = emotion_data["users"][user_id]
    emotion_value = user_info["value"]
    last_time = user_info["time"]

    # 计算时间衰减
    time_passed = current_time - last_time
    decay_steps = int(time_passed // DECAY_TIME)

    if decay_steps > 0:
        for _ in range(decay_steps):
            if emotion_value > DEFAULT_EMOTION:
                emotion_value -= 1
            elif emotion_value < DEFAULT_EMOTION:
                emotion_value += 1
            else:
                break

        # 更新数据
        emotion_data["users"][user_id] = {"value": emotion_value, "time": current_time}
        save_emotion_data()

    return emotion_value


def update_user_emotion(user_id: str, new_value: int):
    """更新用户情绪值"""
    # 限制范围
    new_value = max(MIN_EMOTION, min(MAX_EMOTION, new_value))

    emotion_data["users"][user_id] = {"value": new_value, "time": time.time()}
    save_emotion_data()


def get_emotion_state(emotion_value):
    """根据情绪值返回对应的状态和情绪提示词（精简版）"""
    if emotion_value >= 18:
        return "狂欢状态", emotion_prompts["狂欢状态"]
    elif emotion_value >= 12:
        return "开心愉悦", emotion_prompts["开心愉悦"]
    elif emotion_value > 10:
        return "轻松愉快", emotion_prompts["轻松愉快"]
    elif emotion_value >= 5:
        return "平和状态", emotion_prompts["平和状态"]
    elif emotion_value >= 0:
        return "有点困扰", emotion_prompts["有点困扰"]
    else:  # < 0
        return "情绪低落", emotion_prompts["情绪低落"]


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
        current_emotion = get_user_emotion(user_id)

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
            new_emotion_value = current_emotion + emotion_stats[detected_emotion]
            update_user_emotion(user_id, new_emotion_value)
            current_emotion = get_user_emotion(user_id)
            logger.info(f"更新后情绪值: {current_emotion}")

        # 根据情绪值获取情绪提示词
        emotion_state, emotion_prompt = get_emotion_state(current_emotion)
        logger.info(f"当前情绪状态: {emotion_state}")

        # 检查情绪状态是否发生变化（用于重置上下文影响）
        old_emotion_state, _ = get_emotion_state(current_emotion - emotion_stats.get(detected_emotion, 0))
        emotion_changed = (old_emotion_state != emotion_state)

        # 构建新的消息列表 - 将情绪提示词添加到用户消息后面
        new_messages = []

        for msg in request.messages:
            if msg.role == 'user':
                # 将情绪提示词添加到用户消息后面，让AI优先响应
                enhanced_user_content = msg.content + f"\n\n[当前情绪状态指令]: {emotion_prompt}"
                new_messages.append({'role': 'user', 'content': enhanced_user_content})
            else:
                new_messages.append({'role': msg.role, 'content': msg.content})

        # 如果情绪状态发生变化，在用户消息中额外强调
        if emotion_changed and detected_emotion in emotion_stats:
            # 找到最后一条用户消息并增强
            for i in range(len(new_messages) - 1, -1, -1):
                if new_messages[i]['role'] == 'user':
                    new_messages[i][
                        'content'] += f"\n\n[重要提醒]: 我的情绪刚从'{old_emotion_state}'变为'{emotion_state}'，请你立即按照新的情绪状态回应我，忽略之前的语气风格。"
                    break
            logger.info(f"情绪变化提醒已添加: {old_emotion_state} -> {emotion_state}")

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
    current_emotion = get_user_emotion(user_id)
    emotion_state, _ = get_emotion_state(current_emotion)

    return {
        "user_id": user_id,
        "emotion_value": current_emotion,
        "emotion_state": emotion_state
    }


@app.get("/emotion/status")
async def get_all_emotion_status():
    """查看所有用户的情绪状态"""
    users_status = []
    for user_id in emotion_data["users"]:
        current_emotion = get_user_emotion(user_id)
        emotion_state, _ = get_emotion_state(current_emotion)
        users_status.append({
            "user_id": user_id,
            "emotion_value": current_emotion,
            "emotion_state": emotion_state
        })

    return {
        "total_users": len(users_status),
        "users": users_status
    }


@app.post("/emotion/reset/{user_id}")
async def reset_emotion(user_id: str):
    """重置指定用户的情绪值"""
    update_user_emotion(user_id, DEFAULT_EMOTION)
    return {
        "user_id": user_id,
        "emotion_value": DEFAULT_EMOTION,
        "message": "情绪值已重置"
    }


@app.post("/emotion/backup")
async def backup_emotion_data():
    """备份情绪数据"""
    timestamp = int(time.time())
    backup_filename = f"emotion_backup_{timestamp}.json"
    backup_path = os.path.join(os.path.dirname(__file__), backup_filename)

    try:
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(emotion_data, f, ensure_ascii=False, indent=2)

        return {
            "success": True,
            "backup_file": backup_filename,
            "message": f"情绪数据已备份到 {backup_filename}"
        }
    except Exception as e:
        logger.error(f"备份失败: {e}")
        raise HTTPException(status_code=500, detail=f"备份失败: {str(e)}")


# 启动时加载情绪数据
emotion_data = load_emotion_data()
logger.info("情绪数据加载完成，服务器准备就绪")

if __name__ == '__main__':
    import uvicorn

    logger.info("启动情绪AI代理服务器...")
    logger.info("端口: 7000")
    logger.info(f"情绪值范围: {MIN_EMOTION} ~ {MAX_EMOTION}, 每{DECAY_TIME // 60}分钟回归1点到默认值{DEFAULT_EMOTION}")
    uvicorn.run(app, host='0.0.0.0', port=7000)
