import os
from typing import List,Dict
from pathlib import Path
from importlib.metadata import version
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
# 添加 CORS 中间件以允许跨域请求
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import soundfile as sf
import numpy as np
import io
import wave
# ---------------------------------------------------------------------------
from ai_vtuber import ALL_TOOLS, TOOLS_PACKAGE, agent ,SqlMemory
from ai_vtuber.asr_module import ASR
from ai_vtuber.tts_module import TTSInterface
# ---------------------------------------------------------------------------
# 1. 載入環境變數（如果有 .env 檔會自動讀取）
# ---------------------------------------------------------------------------
load_dotenv()  # .env 放在 workspace 根目錄或此子目錄皆可

# --------------------------------------------------------------------------
# 系統根目錄
ROOT_PATH = Path(__file__).parent

# 必要的路徑與設定，若未提供則使用預設值
if os.getenv("REFERENCE_WAV_PATH"):
    WAV_PATH = str( ROOT_PATH / os.getenv("REFERENCE_WAV_PATH","") )
else:
    WAV_PATH = None
AGENT_MODEL=os.getenv("MODEL","qwen2.5:7b")
PROMPT=os.getenv("PROMPT",None)
ASR_MODEL=os.getenv("ASR_MODEL","")
MAX_MEMORY_ROUNDS=int(os.getenv("MAX_MEMORY_ROUNDS", 20))
MEMORY_ROOT_PATH:Path = Path(os.getenv("MEMORY_ROOT_PATH", (ROOT_PATH / "memory") ) )
INFORMATION_INTEGRATION_MODEL = os.getenv("INFORMATION_INTEGRATION_MODEL", "qwen3:4b-instruct-2507-q4_K_M")


agent_kwargs={}
if PROMPT:
    agent_kwargs["prompt"]=PROMPT
# ---------------------------------------------------------------------------
# 2. 初始化核心資源 – tools, ASR, TTS, Agent
# Initialise shared components -------------------------------------------------
tools = []
asr   = ASR(model_name=ASR_MODEL)                                 # uses default model download logic
tts   = TTSInterface(
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

# Configure CORS to allow any origin (adjust as needed for production)
app.add_middleware(
    CORSMiddleware,
    # 配置 CORS，允許所有來源（可根據生產環境需求調整）
    allow_origins=["*"],  # 允許所有來源
    allow_credentials=True, # 是否允許跨域攜帶憑證（如 Cookie）
    allow_methods=["*"],   # 允許的所有 HTTP 方法（如 GET, POST, OPTIONS 等）
    allow_headers=["*"],   # 允許的 HTTP 標頭
)

class UserAI(BaseModel):
    id: str
    agent: agent
    memory: SqlMemory

agent_pools :Dict[str, UserAI]= {}

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
    #確認是否有身份(沒有就創建或載入)
    if req.user_id not in agent_pools:
        user_dir_path: Path = (MEMORY_ROOT_PATH / req.user_id )
        user_dir_path.mkdir(parents=True,exist_ok=True)
        db_path: Path = user_dir_path / "memory.sqlite"
        memory = SqlMemory(
            db_path = db_path,
            model_name = INFORMATION_INTEGRATION_MODEL
        )
        ai=agent(
            **agent_kwargs,
            model=AGENT_MODEL,
            tools=tools,
            max_memory_rounds=MAX_MEMORY_ROUNDS,
            long_term_memory=memory
        )
        agent_pools[req.user_id] = UserAI(id=req.user_id,agent=ai,memory=memory)

    #推理並回傳
    response = agent_pools[req.user_id].agent.invoke(req.text)
    return {"response": response}

@app.post("/tts")
async def tts_api(req: TTSRequest):
    """文字轉語音接口"""
    # 呼叫 TTS 物件的 tts 方法
    result = tts.tts(req.text,WAV_PATH,play_audio=False)
    samplerate = 24000  # 假設 TTS 模組輸出的取樣率為 24000 Hz
    if result is None:
        print("TTS 生成失敗，請參考以上原因")
        raise HTTPException(status_code=500, detail="TTS 生成失敗")
    # 轉換 ndarray 為 WAV 格式流
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, result, samplerate=samplerate,
             format="WAV", subtype="PCM_16")

    audio_buffer.seek(0)
    return Response(content=audio_buffer.read(), media_type="audio/wav")

@app.post("/asr")
async def asr_api(req: ASRRequest):
    """語音轉文字接口"""
    # 將輸入的 list 轉換為 numpy array 以符合 ASR 模組要求
    audio_np = np.array(req.audio)
    result = asr.speach_to_text(audio=audio_np, samplerate=req.samplerate)
    return {"text": result}