import os
import pathlib
import shutil
from typing import Optional
from yt_dlp import YoutubeDL

class FFmpegNotFoundError(RuntimeError):
    pass

def download_audio(
    url: str,
    out_dir: str = "downloads",
    filename_pattern: str = "%(title)s.%(ext)s",
    ext: str = "mp3",
    bitrate_kbps: int = 192,
) -> Optional[pathlib.Path]:
    """
    Скачивает аудио из видео по URL с YouTube (или поддерживаемого сайта),
    извлекает/конвертирует дорожку через ffmpeg и сохраняет в out_dir.
    
    Аргументы:
      - url: ссылка на видео (например, https://www.youtube.com/watch?v=...).
      - out_dir: папка для сохранения.
      - filename_pattern: шаблон имени файла yt-dlp (оставьте по умолчанию).
      - ext: целевой формат аудио: 'mp3' или 'm4a' и т.п.
      - bitrate_kbps: целевой битрейт для mp3 (игнорируется для некоторых контейнеров).
    
    Возвращает:
      - pathlib.Path к сохраненному файлу или None в случае неудачи.
    """
    # Проверим наличие ffmpeg заранее, чтобы дать понятную ошибку
    if shutil.which("ffmpeg") is None:
        raise FFmpegNotFoundError(
            "ffmpeg не найден в PATH. Установите ffmpeg и повторите попытку."
        )

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Настройки yt-dlp:
    # - bestaudio выбирает лучший аудиопоток
    # - постпроцессоры: извлечь аудио и перекодировать при необходимости
    postprocessors = [{"key": "FFmpegExtractAudio", "preferredcodec": ext}]
    # Для MP3 можно указать битрейт
    if ext.lower() == "mp3":
        postprocessors[0]["preferredquality"] = str(bitrate_kbps)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(out_dir, filename_pattern),
        "quiet": True,          # без лишнего шума в консоли
        "noprogress": True,
        "postprocessors": postprocessors,
        # Улучшение совместимости с YouTube
        "extract_flat": False,
        "ignoreerrors": False,
        "retries": 3,
    }

    # Выполняем загрузку
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # yt-dlp возвращает итоговое имя файла через prepare_filename до постпроцессинга,
            # поэтому вычислим ожидаемое имя с учётом ext.
            temp_path = pathlib.Path(ydl.prepare_filename(info))
            final_path = temp_path.with_suffix(f".{ext}")

            # Иногда у YouTube в названии бывают недопустимые символы,
            # yt-dlp уже нормализует имя, но проверим существование:
            if final_path.exists():
                return final_path.resolve()

            # На случай, если контейнер уже был целевым (например, m4a)
            if temp_path.exists() and temp_path.suffix.lower() == f".{ext.lower()}":
                return temp_path.resolve()

            # Если ничего не нашли, попробуем отыскать по каталогу последний изменённый файл с нужным расширением
            candidates = sorted(
                pathlib.Path(out_dir).glob(f"*.{ext}"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            return candidates[0].resolve() if candidates else None
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return None

if __name__ == "__main__":
    test_url = "https://www.youtube.com/watch?v=8Me3xlXh5Tk"
    directory = "downloads/"
    path = download_audio(test_url, out_dir=directory, ext="mp3")
    print("Сохранено в:", path)