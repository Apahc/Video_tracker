"""
Модуль стабилизации видео через FFmpeg + vidstab.

Использует двухпроходный подход:
1. vidstabdetect — анализ движения → transforms.trf
2. vidstabtransform — применение стабилизации

Особенности:
- Авто-поиск FFmpeg (WinGet, PATH, ручной путь)
- Прогресс-бар по реальному времени обработки
- Безопасное удаление временных файлов
- Подробные логи и защита от падений
- Инструкции по установке FFmpeg в комментариях

ПАРАМЕТРЫ (настраиваемые):
    - fallback_duration_sec: 30.0   → длительность по умолчанию
    - shakiness: 5                  → чувствительность анализа
    - accuracy: 15                  → точность анализа
    - smoothing: 15                 → плавность стабилизации
    - zoom: 5                       → компенсация обрезки (%)
    - optzoom: 2                    → динамический зум
    - crf: 23                       → качество видео
    - preset: "medium"              → скорость/качество

Путь до данного файла: src/stabilize.py
"""
import subprocess
import os
import re
import sys
import glob
from pathlib import Path
from tqdm import tqdm
import shutil

from typing import Optional


class StabilizerConfig:
    """Конфигурация стабилизации"""
    fallback_duration_sec: float = 30.0
    shakiness: int = 5
    accuracy: int = 15
    smoothing: int = 15
    zoom: int = 5
    optzoom: int = 2
    maxshift: int = -1
    crf: int = 23
    preset: str = "medium"
    unsharp: str = "5:5:0.8:3:3:0.4"


def get_ffmpeg_path() -> str:
    """
    Автоматически находит путь к ffmpeg.exe

    Порядок поиска:

    1. shutil.which("ffmpeg") — в PATH
    2. WinGet: Путь до ffmpeg.exe
    3. Ошибка, если не найден

    Returns
    -------
    str
        Полный путь к ffmpeg.exe
    """
    # 1. Проверка PATH
    path = shutil.which("ffmpeg")
    if path and os.path.isfile(path):
        return path

    # 2. Поиск через WinGet (Windows)
    pattern = r"C:\Users\fantasPCmini\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_*\ffmpeg-*\bin\ffmpeg.exe"
    for f in glob.glob(pattern):
        if os.path.isfile(f):
            return f

    raise RuntimeError(
        "FFmpeg не найден!\n\n"
        "УСТАНОВКА FFmpeg (Windows):\n"
        "1. Через WinGet (рекомендуется):\n"
        "   Откройте PowerShell от имени администратора и выполните:\n"
        "       winget install Gyan.FFmpeg\n"
        "   → FFmpeg установится автоматически, и get_ffmpeg_path() найдёт его.\n\n"
        "2. Вручную:\n"
        "   • Скачайте с: https://www.gyan.dev/ffmpeg/builds/\n"
        "   • Выберите 'ffmpeg-git-full.7z'\n"
        "   • Распакуйте в C:\\ffmpeg\n"
        "   • Добавьте в PATH: C:\\ffmpeg\\bin\n"
        "   • Или раскомментируйте 'manual_path' в коде выше.\n\n"
        "После установки перезапустите IDE/терминал."
    )


def stabilize_video_ffmpeg(
        input_path: str,
        output_path: str,
        config: Optional[StabilizerConfig] = None
) -> str:
    """
    Стабилизация видео через FFmpeg + vidstab

    Parameters
    ----------
    input_path : str
        Путь к исходному видео (MP4, AVI и т.д.)
    output_path : str
        Путь к стабилизированному видео
    config : StabilizerConfig, optional
        Настройки

    Returns
    -------
    str
        Путь к стабилизированному видео (или исходному при ошибке)
    """
    if config is None:
        config = StabilizerConfig()

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = get_ffmpeg_path()
    print(f"FFmpeg найден: {ffmpeg}")

    # Получаем длительность видео
    print("Получение длительности видео...")
    duration_cmd = [ffmpeg, "-i", str(input_path)]
    result = subprocess.run(duration_cmd, capture_output=True, text=True)

    duration_match = re.search(r"Duration: (\d{2}):(\d{2}):(\d{2}\.\d{2})", result.stderr)
    total_seconds = config.fallback_duration_sec
    if duration_match:
        h, m, s = map(float, duration_match.groups())
        total_seconds = h * 3600 + m * 60 + s
    print(f"Длительность видео: {total_seconds:.2f} сек")

    # 1. Анализ движения (vidstabdetect)
    trf_path = output_path.parent / "transforms.trf"
    if trf_path.exists():
        try:
            trf_path.unlink()
        except Exception():
            pass

    print("FFmpeg: Анализ движения (vidstabdetect)...")
    detect_cmd = [
        ffmpeg, "-i", str(input_path),
        "-vf", "vidstabdetect=result=transforms.trf",
        "-f", "null", "-"
    ]
    subprocess.run(detect_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


    if not trf_path.exists():
        print("ОШИБКА: transforms.trf НЕ создан!")
        return str(input_path)
    print(f"transforms.trf создан: {os.path.getsize(trf_path)} байт")

    # 2. Применение стабилизации (vidstabtransform)
    print("FFmpeg: Применение стабилизации (vidstabtransform)...")

    transform_cmd = [
        ffmpeg, "-i", str(input_path),
        "-vf", (
            f"vidstabtransform=input={trf_path}"
            f":smoothing={config.smoothing}"                # плавность
            f":zoom={config.zoom}"                          # компенсация обрезки
            f":optzoom={config.optzoom}"                    # динамический зум
            f":maxshift={config.maxshift}"                  # без ограничения
            f",unsharp={config.unsharp}"
        ),
        "-c:v", "libx264",
        "-crf", str(config.crf),                            # качество
        "-preset", config.preset,                           # баланс скорости/качества
        "-c:a", "aac",                                      # аудио
        "-b:a", "192k",
        "-y",                                               # перезаписать
        str(output_path)
    ]

    # Запуск с прогресс-баром
    process = subprocess.Popen(
        transform_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    pbar = tqdm(total=total_seconds, unit="s", desc="Стабилизация", leave=True)
    current_time = 0.0

    for line in process.stdout:
        sys.stdout.flush()
        # FFmpeg пишет прогресс в stdout
        if "time=" in line:
            match = re.search(r"time=(\d{2}):(\d{2}):(\d{2}\.\d{2})", line)
            if match:
                h, m, s = map(float, match.groups())
                current_time = h * 3600 + m * 60 + s
                pbar.n = min(current_time, total_seconds)
                pbar.refresh()

    process.wait()
    pbar.close()

    if process.returncode != 0:
        print(f"ОШИБКА стабилизации (код {process.returncode})")
        return str(input_path)

    # Очистка
    try:
        os.remove(trf_path)
    except Exception():
        pass

    print(f"Стабилизация завершена: {output_path}")
    return str(output_path)
