import ffmpeg
import whisper
import os
import sys
import time
import requests
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# API 키 리스트
API_KEYS = os.getenv("DEEPL_API_KEYS").split(",")
using_key_index = 0  # 현재 사용 중인 API 키 인덱스

def extract_audio(mp4_file, audio_file):
    """MP4 파일에서 오디오 추출"""
    print(f"오디오 추출 시작: {mp4_file}")
    ffmpeg.input(mp4_file).output(audio_file, acodec='mp3').run(overwrite_output=True)

def audio_to_srt(audio_file, srt_file):
    """Whisper 모델을 사용하여 오디오를 SRT 파일로 변환"""
    model = whisper.load_model("medium")  # MPS(GPU) 사용 가능
    
    print(f"텍스트 추출 시작: {audio_file}")
    result = model.transcribe(audio_file, fp16=False)
    
    print(f"SRT 생성 시작: {srt_file}")
    with open(srt_file, "w", encoding="utf-8") as f_srt:
        for i, segment in enumerate(result["segments"], start=1):
            start = format_time(segment["start"])
            end = format_time(segment["end"])
            text = segment["text"]
            
            f_srt.write(f"{i}\n{start} --> {end}\n{text}\n\n") 

def translate_text(text):
    """DeepL API를 사용하여 텍스트를 한국어로 번역 (순차적 API 키 사용)"""
    global using_key_index
    API_URL = "https://api-free.deepl.com/v2/translate"
    
    while using_key_index < len(API_KEYS):
        api_key = API_KEYS[using_key_index]
        params = {
            "auth_key": api_key,
            "text": text,
            "target_lang": "KO"  # 한국어 번역
        }
        response = requests.post(API_URL, data=params)
        
        if response.status_code == 200:
            return response.json()["translations"][0]["text"]
        else:
            using_key_index += 1  # 실패하면 다음 키 사용
            print(f"번역 실패: {response.status_code} - {response.text}, 다음 API 키 사용 {using_key_index}")
    
    print("모든 API 키 사용 실패")
    sys.exit(1)
    

def srt_to_translated_srt(srt_file, translated_srt_file):
    """SRT 파일을 읽어 한국어 번역본 생성"""
    print(f"한글 번역 시작: {srt_file}")
    
    with open(srt_file, "r", encoding="utf-8") as f_srt, open(translated_srt_file, "w", encoding="utf-8") as f_translated:
        lines = f_srt.readlines()
        
        for line in lines:
            if "-->" in line or line.strip().isdigit() or line.strip() == "":
                f_translated.write(line)
            else:
                translated_text = translate_text(line.strip())
                f_translated.write(f"{translated_text}\n")

def format_time(seconds):
    """시간을 SRT 형식으로 변환"""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"

if __name__ == "__main__":
    source_folder = "source"
    mp3_folder = "mp3"
    srt_folder = "srt"
    os.makedirs(source_folder, exist_ok=True)
    os.makedirs(mp3_folder, exist_ok=True)
    os.makedirs(srt_folder, exist_ok=True)
    
    # Step 1: MP4 -> MP3 변환
    print("MP4 -> MP3 변환 시작")
    for file in os.listdir(source_folder):
        if file.endswith(".mp4") and not os.path.exists(os.path.join(mp3_folder, f"{os.path.splitext(file)[0]}.mp3")):
            mp4_file = os.path.join(source_folder, file)
            audio_file = os.path.join(mp3_folder, f"{os.path.splitext(file)[0]}.mp3")
            extract_audio(mp4_file, audio_file)
    print("MP4 -> MP3 변환 완료")
    
    # Step 2: MP3 -> SRT 변환
    print("MP3 -> SRT 변환 시작")
    for file in os.listdir(mp3_folder):
        if file.endswith(".mp3") and not os.path.exists(os.path.join(srt_folder, f"{os.path.splitext(file)[0]}.srt")):
            audio_file = os.path.join(mp3_folder, file)
            srt_file = os.path.join(srt_folder, f"{os.path.splitext(file)[0]}.srt")
            audio_to_srt(audio_file, srt_file)
    print("MP3 -> SRT 변환 완료")
    
    # Step 3: SRT -> 번역된 SRT(ko.srt) 변환
    print("SRT -> 한글 번역 시작")
    for file in os.listdir(srt_folder):
        if file.endswith(".srt") and not file.endswith(".ko.srt") and not os.path.exists(os.path.join(srt_folder, f"{os.path.splitext(file)[0]}.ko.srt")):
            srt_file = os.path.join(srt_folder, file)
            translated_srt_file = os.path.join(srt_folder, f"{os.path.splitext(file)[0]}.ko.srt")
            srt_to_translated_srt(srt_file, translated_srt_file)
    print("SRT -> 한글 번역 완료")
