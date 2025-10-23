# faster_whisper_transcribe.py

"""
Faster-Whisper использует CTranslate2 - оптимизированный движок для CPU.
Установка: pip install faster-whisper
"""

from faster_whisper import WhisperModel
import time
from pathlib import Path


def transcribe_with_faster_whisper(
    audio_path: str,
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8"  # int8 оптимален для CPU
) -> dict:
    """
    Транскрибация аудио с помощью Faster-Whisper
    
    Args:
        audio_path: путь к аудиофайлу
        model_size: размер модели (tiny, base, small, medium, large-v2, large-v3)
        device: cpu или cuda
        compute_type: int8, int16, float16, float32 (int8 быстрее на CPU)
    
    Returns:
        dict с результатами транскрипции и метриками
    """
    
    print(f"[Faster-Whisper] Загрузка модели '{model_size}' на {device}...")
    start_load = time.time()
    
    # cpu_threads - количество потоков для CPU (по умолчанию использует все доступные)
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=4,  # можете настроить под свою систему
        num_workers=1,
        download_root="models"
    )
    
    load_time = time.time() - start_load
    print(f"[Faster-Whisper] Модель загружена за {load_time:.2f} сек")
    
    print(f"[Faster-Whisper] Начинаем транскрипцию: {audio_path}")
    start_transcribe = time.time()
    
    # Транскрибация с настройками для русского языка
    segments, info = model.transcribe(
        audio_path,
        language="ru",
        beam_size=5,
        vad_filter=True,  # Voice Activity Detection - фильтрует тишину
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    # Собираем текст из сегментов
    full_text = ""
    segments_list = []
    
    for segment in segments:
        full_text += segment.text + " "
        segments_list.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text
        })
    
    transcribe_time = time.time() - start_transcribe
    
    print(f"[Faster-Whisper] Транскрипция завершена за {transcribe_time:.2f} сек")
    print(f"[Faster-Whisper] Обнаружен язык: {info.language} (вероятность: {info.language_probability:.2f})")
    
    return {
        "text": full_text.strip(),
        "segments": segments_list,
        "load_time": load_time,
        "transcribe_time": transcribe_time,
        "total_time": load_time + transcribe_time,
        "language": info.language,
        "language_probability": info.language_probability
    }


if __name__ == "__main__":
    # Пример использования
    audio_file = "downloads/output.mp3"  # Замените на ваш файл
    
    # Для CPU лучше использовать модели tiny, base или small
    result = transcribe_with_faster_whisper(
        audio_file,
        model_size="base",  # base - хороший баланс скорости и качества
        device="cpu",
        compute_type="int8"
    )
    
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТ:")
    print("="*50)
    print(result["text"])
    print(f"\nВремя загрузки: {result['load_time']:.2f} сек")
    print(f"Время транскрипции: {result['transcribe_time']:.2f} сек")
    print(f"Общее время: {result['total_time']:.2f} сек")