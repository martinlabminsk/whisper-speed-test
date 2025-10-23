import whisper
import time
from pathlib import Path
import torch


def transcribe_with_whisper(
    audio_path: str,
    model_size: str = "base",
    model_path: str = None,  # Добавляем параметр для пути к модели
    device: str = None
) -> dict:
    """
    Транскрибация аудио с помощью оригинального Whisper
    
    Args:
        audio_path: путь к аудиофайлу
        model_size: размер модели (tiny, base, small, medium, large)
        model_path: путь к файлу модели (.pt). Если указан, игнорирует model_size
        device: None (автовыбор), "cpu" или "cuda"
    
    Returns:
        dict с результатами транскрипции и метриками
    """
    
    # Автоопределение устройства
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Определяем, что использовать: путь к модели или размер
    model_identifier = model_path if model_path else model_size
    
    print(f"[Whisper] Загрузка модели '{model_identifier}' на {device}...")
    start_load = time.time()
    
    model = whisper.load_model(model_identifier, device=device)
    
    load_time = time.time() - start_load
    print(f"[Whisper] Модель загружена за {load_time:.2f} сек")
    
    print(f"[Whisper] Начинаем транскрипцию: {audio_path}")
    start_transcribe = time.time()
    
    # Транскрибация с настройками для русского языка
    result = model.transcribe(
        audio_path,
        language="ru",
        verbose=False,
        fp16=False  # fp16=False для CPU (на GPU можно True)
    )
    
    transcribe_time = time.time() - start_transcribe
    
    print(f"[Whisper] Транскрипция завершена за {transcribe_time:.2f} сек")
    print(f"[Whisper] Обнаружен язык: {result['language']}")
    
    return {
        "text": result["text"].strip(),
        "segments": result["segments"],
        "load_time": load_time,
        "transcribe_time": transcribe_time,
        "total_time": load_time + transcribe_time,
        "language": result["language"]
    }


if __name__ == "__main__":
    # Пример использования
    audio_file = "downloads/output.mp3"  # Замените на ваш файл
    
    # Вариант 1: Использование предустановленной модели по имени
    result = transcribe_with_whisper(
        audio_file,
        model_size="base",
        device="cpu"
    )
    
    # Вариант 2: Использование локальной модели по пути
    # local_model_path = "/path/to/your/local/model.pt"  # Укажите путь к вашей модели
    # result = transcribe_with_whisper(
    #     audio_file,
    #     model_path=local_model_path,
    #     device="cpu"
    # )
    
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТ:")
    print("="*50)
    print(result["text"])
    print(f"\nВремя загрузки: {result['load_time']:.2f} сек")
    print(f"Время транскрипции: {result['transcribe_time']:.2f} сек")
    print(f"Общее время: {result['total_time']:.2f} сек")