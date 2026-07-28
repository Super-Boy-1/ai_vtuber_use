# ai_vtuber_use

這個專案將 `ai_vtuber` 套件包裝成一個簡單的 FastAPI 後端服務，提供：

- `/chat`：LLM 聊天 Agent
- `/tts`：文字轉語音
- `/asr`：語音轉文字

## 目標

讓你能夠快速部署一個可對外呼叫的 AI VTuber 後端 API，並在本地或雲端環境中使用。

## 先決條件

- Python 3.11+（或專案所需版本）
- 已建立虛擬環境
- 安裝依賴：

```bash
pip install -r requirements.txt
```

## 環境設定

專案會從 `.env` 讀取設定。常見變數：

```env
MODEL
ASR_MODEL
PROMPT
REFERENCE_WAV_PATH
MAX_MEMORY_ROUNDS
```

如果沒有 `.env`，程式會使用預設值。

## 部署方式

### 1. 本地測試

在本機啟動 Uvicorn：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

如果你想讓其他裝置可連線，請使用 `--host 0.0.0.0`。

### 3. 正式部署建議

- 使用反向代理（如 Nginx）做 HTTPS 與路由轉發
- 加上 CORS 與身份驗證保護 API
- 避免直接暴露 `chat` 路徑給公開網路
- 若 `agent_pools` 可能過多，考慮加入使用者 session 清理機制

## API 使用說明

以下為可用的三個 POST API，示範使用 Python `requests` 呼叫。

### 1. `/chat`

```python
import requests

url = "http://localhost:8000/chat"
payload = {"user_id": "user1", "text": "你好"}

resp = requests.post(url, json=payload)
print(resp.json())
```

範例回傳：

```json
{"response": "..."}
```

### 2. `/tts`

```python
import requests

url = "http://localhost:8000/tts"
payload = {"text": "這是一段測試文字"}

resp = requests.post(url, json=payload)
print(resp.json())
```

回傳格式：

```json
{"audio": [0.0, 0.1, ...]}
```

### 3. `/asr`

```python
import requests

url = "http://localhost:8000/asr"
payload = {
    "audio": [0.0, 0.1, ...],
    "samplerate": 16000
}

resp = requests.post(url, json=payload)
print(resp.json())
```

回傳格式：

```json
{"text": "辨識結果文字"}
```

## 注意事項

- API 目前接收 `audio` 為 JSON float list。若要改成檔案上傳，需調整 `ASRRequest` 與路由實作。
- `agent_pools` 會依 `user_id` 建立 agent。請確保 `user_id` 不會無限制增加，避免記憶體耗盡。
- 若使用 GPU，請確認 `TTSInterface` 與 ASR 模型可以正確載入。

## 參考

- 原始專案： https://github.com/Super-Boy-1/ai_vtuber.git
