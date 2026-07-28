import os
from typing import List,Dict
from importlib.metadata import version
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import numpy as np
# ---------------------------------------------------------------------------
from ai_vtuber import ALL_TOOLS, TOOLS_PACKAGE, agent
from ai_vtuber.asr_module import ASR
from ai_vtuber.tts_module import TTSInterface
# ---------------------------------------------------------------------------
# 1. 載入環境變數（如果有 .env 檔會自動讀取）
# ---------------------------------------------------------------------------
load_dotenv()  # .env 放在 workspace 根目錄或此子目錄皆可

# 必要的路徑與設定，若未提供則使用預設值
WAV_PATH = os.getenv("REFERENCE_WAV_PATH","")
AGENT_MODEL=os.getenv("MODEL","qwen2.5:7b")
PROMPT=os.getenv("PROMPT",None)
ASR_MODEL=os.getenv("ASR_MODEL","")
MAX_MEMORY_ROUNDS=int(os.getenv("MAX_MEMORY_ROUNDS", 20))
agent_kwargs={}
if PROMPT:
    agent_kwargs["prompt"]=PROMPT
# ---------------------------------------------------------------------------
# 2. 初始化核心資源 – tools, ASR, TTS, Agent
# Initialise shared components -------------------------------------------------
tools = []
asr   = ASR(model_name=ASR_MODEL)                                 # uses default model download logic
tts   = TTSInterface(
    reference_wav_path=WAV_PATH,
    cuda=True,
    )# enable GPU if available

# ---------------------------------------------------------------------------
# 3. FastAPI 應用程式設定
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI VTuber 後端伺服器",
    version=version('ai_vtuber'),
    description=(
        "提供工具、語音辨識、文字轉語音與 LLM Agent 的 API，\n"
        "所有核心物件在啟動時已載入，可直接於路由中呼叫。"
    ),
)

agent_pools :Dict[str, agent]= {}

# 並行處理的類別定義
class ChatRequest(BaseModel):
    user_id: str
    text: str

class TTSRequest(BaseModel):
    text: str

class ASRRequest(BaseModel):
    audio: List[float]  # 用於接收從 JSON 傳入的音訊數值
    samplerate: int

@app.post("/chat")
async def chat_api(req: ChatRequest):
    """AI Agent 聊天接口"""
    if req.user_id not in agent_pools:
        agent_pools[req.user_id] = agent(
            **agent_kwargs,
            model=AGENT_MODEL,
            tools=tools,
            max_memory_rounds=MAX_MEMORY_ROUNDS,
        )
    response = agent_pools[req.user_id].invoke(req.text)
    return {"response": response}

@app.post("/tts")
async def tts_api(req: TTSRequest):
    """文字轉語音接口"""
    # 呼叫 TTS 物件的 tts 方法
    result = tts.tts(req.text,play_audio=False)
    return {"audio": result}

@app.post("/asr")
async def asr_api(req: ASRRequest):
    """語音轉文字接口"""
    # 將輸入的 list 轉換為 numpy array 以符合 ASR 模組要求
    audio_np = np.array(req.audio)
    result = asr.speach_to_text(audio=audio_np, samplerate=req.samplerate)
    return {"text": result}