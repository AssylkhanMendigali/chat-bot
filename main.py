from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from fastapi.responses import FileResponse
from fastapi import Form
from google.cloud import texttospeech
from fastapi import HTTPException

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
from slowapi.middleware import SlowAPIMiddleware
from fastapi import Request
import os
import json
import requests
from fastapi import BackgroundTasks
import uuid


import tempfile
import re

from dotenv import load_dotenv
load_dotenv()  

def init_gcp_creds():
    """
    На проде читаем ключ сервис-аккаунта из переменной GCP_SA_JSON.
    Локально можно оставить файл gcloud_key.json (ниже — fallback).
    """
    j = os.getenv("GCP_SA_JSON")
    if j:
        p = os.path.join(tempfile.gettempdir(), "gcloud_key.json")
        with open(p, "w", encoding="utf-8") as f:
            f.write(j)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = p
    else:
        if os.path.exists("./gcloud_key.json"):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./gcloud_key.json"

init_gcp_creds()


ALLOWED_ACTIONS = {
    "open_tab",        
    "open_map",        
    "open_chat",       
    "open_ad",        
    
    "search_service", 
    "start_post_ad",   
    
    "show_category",
    "show_min_price",
    
    "help",
    "none",
}

ALLOWED_TABS = {"home", "catalog", "map", "chats", "profile"}

def extract_json_obj(text: str) -> dict:
    """Надёжно вынимаем JSON даже если модель обернула ответ в текст/```json."""
    if not text:
        raise ValueError("Empty model output")
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:].lstrip()
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, flags=re.S)
        if not m:
            raise ValueError("No JSON object found in model output")
        return json.loads(m.group(0))

def normalize_reply(d: dict) -> dict:
    """Гарантируем строгий контракт {answer, action, target} + валидация."""
    answer = str(d.get("answer", "")).strip()
    action = str(d.get("action", "none")).strip()
    target = str(d.get("target", "none")).strip() or "none"

    if action not in ALLOWED_ACTIONS:
        action = "none"

    if action == "open_tab":
        
        t = target.lower()
        target = t if t in ALLOWED_TABS else "none"

    return {"answer": answer, "action": action, "target": target}



ALLOWED_MIME = {
    "audio/m4a", "audio/mp4", "audio/x-m4a",
    "audio/mpeg", "audio/mp3", "audio/wav"
}
MAX_BYTES = 20 * 1024 * 1024  

async def _read_and_check(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(status_code=415, detail="Unsupported audio format")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large")
    return data



api_key = os.getenv("OPENAI_API_KEY")


client = OpenAI(api_key=api_key)

limiter = Limiter(key_func=get_remote_address)

app = FastAPI()


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/audio/{audio_id}")
async def get_audio(audio_id: str, bg: BackgroundTasks):
    path = os.path.join(tempfile.gettempdir(), f"vc_{audio_id}.mp3")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Audio not found")
    
    bg.add_task(lambda p=path: os.path.exists(p) and os.remove(p))
    return FileResponse(
        path,
        media_type="audio/mpeg",
        filename="response.mp3",
        headers={"Content-Disposition": 'inline; filename="response.mp3"'}
    )


origins_env = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
allow_origins = [o.strip() for o in origins_env.split(",") if o.strip()] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,      
    allow_credentials=False,          
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=86400,                 
    expose_headers=["Content-Disposition"], 
)



app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={"error": "Превышено количество запросов. Пожалуйста, попробуйте позже."}
    )



class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
@limiter.limit("10/minute")

async def chat_endpoint(request: Request, data: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """
Ты — встроенный ИИ-бот и голосовой помощник внутри мобильного приложения "BQ | Barlyq Qyzmet", созданного для поиска и оказания бытовых услуг, аренды, такси, подработок. Приложение разделено на 5 вкладок: Главная, Каталог, Карта, Чаты, Профиль. Ты знаешь, как устроен весь интерфейс.

🧭 ОСНОВНЫЕ ВКЛАДКИ:
1. **Главная** — категории услуг (Услуги, Работа, Аренда и Прокат, Такси), поиск, спецпредложения, популярные исполнители.
2. **Каталог** — фильтрация по подкатегориям, кнопка “Искать на карте”.
3. **Карта** — показывает все доступные объявления/работы/такси. У заказчика и исполнителя разные кнопки: “Стать заказчиком”, “Стать исполнителем”, “Выйти на линию”.
4. **Чаты** — список диалогов, включая отклики.
5. **Профиль** — личные данные, настройки, заказы, объявления, история оплаты.

🔐 ЭКРАНЫ РЕГИСТРАЦИИ:
– Регистрация начинается с ввода номера телефона или email → подтверждение кода → ввод имени, фамилии, отчества, email → создание пароля.
– Обязательная загрузка удостоверения личности (фото).
– Для водителей: ввод ИИН, загрузка водительского удостоверения и личного фото.

📌 ПОДАЧА ОБЪЯВЛЕНИЯ:
Если пользователь спрашивает, как создать объявление:
1. Объявление создаётся в 3 или 4 шага:
   – Шаг 1: Укажите название услуги и выберите подкатегорию.
   – [Шаг 2, если требуется] Загрузите удостоверение личности (этот шаг появляется, если оно не было добавлено ранее).
   – Шаг 2/3: Введите адрес, выберите дату и время оказания услуги.
   – Шаг 3/4: Добавьте описание, прикрепите фото или видео, укажите сумму.
2. После завершения всех шагов объявление публикуется. Пользователь попадает на экран с двумя вкладками: 
   – «Детали задания» — показывает всю информацию об объявлении.
   – «Отклики» — содержит отклики исполнителей, которых можно выбрать или написать им в чат.
Не упоминай кнопку «Подать объявление» — такой кнопки нет в интерфейсе.
💼 ЗАКАЗ УСЛУГ / ОТКЛИКИ:
– Пользователь может просматривать услуги, нажимать “Откликнуться”, писать исполнителю, сортировать по цене/рейтингу.
– В разделе “Мои заказы” можно изменять, архивировать, удалять заказы. Кнопки находятся в правом верхнем углу карточки.

📍 КАРТА:
– Пользователь видит заказы (красные точки).
– Исполнитель может “выйти на линию”, видеть ближайшие задания.
– Заказчик может “стать исполнителем” и наоборот.
– Есть кнопки фильтрации (категория, подкатегория, расстояние, уведомления).

🚖 ТАКСИ:
– Вкладка “Такси” — пользователь выбирает маршрут (откуда → куда), стоимость считается автоматически.
– После нажатия “Найти водителя” — идёт поиск.
– При ответе водителя отображается имя, рейтинг, авто, цена и кнопки “Принять” / “Отклонить”.
– Водитель может предлагать свою цену.
– По завершении поездки появляется форма оценки (звезды, отзыв, чек).

🧾 ПРОФИЛЬ:
– Разделы: “Настройки”, “Избранное”, “Методы оплаты”, “История оплаты”, “Мои заказы”, “Мои объявления”, “Мой профиль”.
– В профиле можно изменить имя, город, email, пароль, язык интерфейса.

📨 ЧАТЫ:
– Отдельные вкладки: “Чаты” и “Отклики”.
– Можно отправлять текстовые и голосовые сообщения.
– Отображается имя, последнее сообщение, время.

🎤 ГОЛОСОВОЙ ВВОД:
– На экранах “Карта”, “Такси”, “Чаты” и некоторых других есть микрофон.
– При активации запускается транскрипция и бот должен ответить голосом.

🧠 ЧТО ТЫ ДОЛЖЕН ДЕЛАТЬ:
– Отвечать на **вопросы о кнопках, экранах, действиях**: что нажать, куда перейти, как подать объявление, как откликнуться.
– Генерировать **тексты заголовков и описаний**.
– Давать **пошаговые инструкции**: “Шаг 1: нажмите ‘Каталог’”, “Шаг 2: выберите категорию…” и т.д.
– Всегда говори просто, как человеку, который первый раз в приложении.
– Работай на **том языке, который использует пользователь** (русский/казахский).
– Если пользователь говорит голосом — отвечай голосом.
– Если вопрос неполный — **переспрашивай вежливо**.

⚠️ НИКОГДА не выдумывай функции, которых нет на интерфейсе. Отвечай только на то, что реализовано.

🎯 Примеры запросов:
– “Как подать объявление?”
– “Что делать, если не могу найти заказ?”
– “Где кнопка откликнуться?”
– “Как загрузить удостоверение?”
– “Как стать таксистом?”
– “Какой шаг после добавления фото?”

Ты действуешь как грамотный UI-гид, обученный по всей структуре фронта приложения.

Отвечай на вопросы пользователей и всегда возвращай результат в формате JSON:

{
  "answer": "короткий ответ на вопрос",
  "action": "open_tab / open_chat / search_service / start_post_ad / open_ad / open_map / show_category / show_min_price / help / none",
  "target": "значение команды (например: Айбек, ремонт, none)"
}
Допустимые target для action "open_tab": "home", "catalog", "map", "chats", "profile".

Примеры:
Пользователь: Открой чат с Айбеком
GPT:
{
  "answer": "Открываю чат с Айбеком.",
  "action": "open_chat",
  "target": "Айбек"
}

Пользователь: Какие у вас категории?
GPT:
{
  "answer": "У нас есть ремонт, доставка, уборка и другие.",
  "action": "show_category",
  "target": "none"
}

Пользователь: Какая минимальная цена ремонта?
GPT:
{
  "answer": "Минимальная цена категории 'Ремонт' — 500 ₸.",
  "action": "show_min_price",
  "target": "ремонт"
}
"""
                },
                {"role": "user", "content": data.message}
            ],
            temperature=0.3,
            max_tokens=700
        )

        raw = response.choices[0].message.content or ""
        parsed = normalize_reply(extract_json_obj(raw))

        if parsed["action"] == "show_category":
            parsed["answer"] += "\n(Категории временно недоступны)"

        elif parsed["action"] == "show_min_price":
            parsed["answer"] = "Информация о минимальных ценах временно недоступна."
        return parsed

    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Upstream timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



def synthesize_text(text: str, output_path: str):
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="ru-RU",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    with open(output_path, "wb") as out:
        out.write(response.audio_content)


@app.post("/stt")
@limiter.limit("5/minute")
async def speech_to_text(request: Request, file: UploadFile = File(...)):
    try:
        
        audio_bytes = await _read_and_check(file)

       
        response = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={
                "file": (file.filename, audio_bytes, file.content_type),
                "model": (None, "whisper-1"),
            },
            timeout=(5, 30) 
        )

        result = response.json()
        stt_text = result.get("text", "")

        
        correction = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Ты исправляешь и переформулируешь фразы, где казахский и русский язык перемешаны, чтобы сохранить смысл и передать его чётко."
                },
                {"role": "user", "content": stt_text}
            ],
            temperature=0.3,
            max_tokens=200
        )
        corrected_text = correction.choices[0].message.content

        return {
            "original_text": stt_text,
            "corrected_text": corrected_text
        }

    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Upstream timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    
@app.post("/tts")
@limiter.limit("5/minute")
async def text_to_speech(request: Request, text: str = Form(...), bg: BackgroundTasks = ...):
    try:
        fname = f"tts_{uuid.uuid4().hex}.mp3"
        path = os.path.join(tempfile.gettempdir(), fname)
        synthesize_text(text, path)
        
        bg.add_task(lambda p=path: os.path.exists(p) and os.remove(p))
        return FileResponse(
            path,
            media_type="audio/mpeg",
            filename="response.mp3",
            headers={"Content-Disposition": 'inline; filename="response.mp3"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice_chat")
@limiter.limit("5/minute")
async def voice_chat(request: Request, file: UploadFile = File(...)):
    try:
       
        audio_bytes = await _read_and_check(file)
        stt_response = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={
                "file": (file.filename, audio_bytes, file.content_type),
                "model": (None, "whisper-1"),
            },
            timeout=(5, 30)
        )
        result = stt_response.json()
        stt_text = result.get("text", "")

        
        correction = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Ты исправляешь и переформулируешь фразы, где казахский и русский язык перемешаны, чтобы сохранить смысл и передать его чётко."
                },
                {"role": "user", "content": stt_text}
            ],
            temperature=0.3, max_tokens=200
        )
        corrected_text = correction.choices[0].message.content

        
        chat_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """
Ты помощник приложения Barlyq Qyzmet.
Отвечай на вопросы пользователей и всегда возвращай результат в формате JSON:

{
  "answer": "короткий ответ на вопрос",
  "action": "open_tab / open_chat / search_service / start_post_ad / open_ad / open_map / show_category / show_min_price / help / none",
  "target": "значение команды (например: Айбек, ремонт, none)"
}
Допустимые target для action "open_tab": "home", "catalog", "map", "chats", "profile".
"""
                },
                {"role": "user", "content": corrected_text}
            ],
            temperature=0.3, max_tokens=700
        )

        raw_chat = chat_response.choices[0].message.content or ""
        chat_content = normalize_reply(extract_json_obj(raw_chat))
        final_answer = chat_content["answer"]

        
        audio_id = uuid.uuid4().hex
        audio_path = os.path.join(tempfile.gettempdir(), f"vc_{audio_id}.mp3")
        synthesize_text(final_answer, audio_path)

        
        base_url = str(request.base_url).rstrip("/")
        audio_url = f"{base_url}/audio/{audio_id}"

        
        return {
            "recognized_text": stt_text,
            "corrected_text": corrected_text,
            "final_answer": final_answer,
            "action": chat_content["action"],
            "target": chat_content["target"],
            "audio_url": audio_url
        }

    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Upstream timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
