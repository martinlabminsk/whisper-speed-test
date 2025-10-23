# benchmark_module.py

"""
Модуль для сравнительной оценки Whisper и Faster-Whisper
Включает замеры времени, памяти и метрики точности (WER, CER)
"""

import time
import psutil
import os
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass
import jiwer  # pip install jiwer
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Результаты бенчмарка"""
    model_name: str
    load_time: float
    transcribe_time: float
    total_time: float
    memory_used_mb: float
    cpu_percent: float
    transcription: str
    wer: Optional[float] = None  # Word Error Rate
    cer: Optional[float] = None  # Character Error Rate


class ASRBenchmark:
    """Класс для бенчмаркинга ASR систем"""
    
    def __init__(self, audio_path: str, reference_text: Optional[str] = None):
        """
        Args:
            audio_path: путь к аудиофайлу для тестирования
            reference_text: эталонный текст для расчета WER/CER (опционально)
        """
        self.audio_path = audio_path
        self.reference_text = reference_text
        self.results: List[BenchmarkResult] = []
    
    def benchmark_function(
        self,
        transcribe_func: Callable,
        model_name: str,
        **kwargs
    ) -> BenchmarkResult:
        """
        Замер производительности функции транскрипции
        
        Args:
            transcribe_func: функция транскрипции
            model_name: название модели для отображения
            **kwargs: дополнительные параметры для функции
        
        Returns:
            BenchmarkResult с метриками
        """
        
        # Запоминаем начальное состояние памяти
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"\n{'='*60}")
        print(f"Бенчмарк: {model_name}")
        print(f"{'='*60}")
        
        # Запускаем транскрипцию
        start_cpu = time.time()
        cpu_percent_start = psutil.cpu_percent(interval=None)
        
        result = transcribe_func(self.audio_path, **kwargs)
        
        cpu_percent_end = psutil.cpu_percent(interval=None)
        
        # Замеряем память после
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = mem_after - mem_before
        
        # Вычисляем метрики точности, если есть эталонный текст
        wer, cer = None, None
        if self.reference_text:
            wer = self.calculate_wer(self.reference_text, result["text"])
            cer = self.calculate_cer(self.reference_text, result["text"])
            print(f"\n📊 Метрики точности:")
            print(f"   WER (Word Error Rate): {wer:.2%}")
            print(f"   CER (Character Error Rate): {cer:.2%}")
        
        benchmark_result = BenchmarkResult(
            model_name=model_name,
            load_time=result["load_time"],
            transcribe_time=result["transcribe_time"],
            total_time=result["total_time"],
            memory_used_mb=memory_used,
            cpu_percent=(cpu_percent_start + cpu_percent_end) / 2,
            transcription=result["text"],
            wer=wer,
            cer=cer
        )
        
        self.results.append(benchmark_result)
        
        print(f"\n⏱️  Производительность:")
        print(f"   Загрузка модели: {benchmark_result.load_time:.2f} сек")
        print(f"   Транскрипция: {benchmark_result.transcribe_time:.2f} сек")
        print(f"   Общее время: {benchmark_result.total_time:.2f} сек")
        print(f"   Использовано памяти: {benchmark_result.memory_used_mb:.1f} MB")
        print(f"   Загрузка CPU: {benchmark_result.cpu_percent:.1f}%")
        
        return benchmark_result
    
    @staticmethod
    def calculate_wer(reference: str, hypothesis: str) -> float:
        """Вычисление Word Error Rate"""
        return jiwer.wer(reference, hypothesis)
    
    @staticmethod
    def calculate_cer(reference: str, hypothesis: str) -> float:
        """Вычисление Character Error Rate"""
        return jiwer.cer(reference, hypothesis)
    
    def print_comparison(self):
        """Печать сравнительной таблицы результатов"""
        if len(self.results) < 2:
            print("Недостаточно результатов для сравнения")
            return
        
        print(f"\n{'='*80}")
        print("📊 СРАВНИТЕЛЬНЫЙ АНАЛИЗ")
        print(f"{'='*80}\n")
        
        for i, result in enumerate(self.results, 1):
            print(f"{i}. {result.model_name}")
            print(f"   Время: {result.total_time:.2f}с (загрузка: {result.load_time:.2f}с, транскрипция: {result.transcribe_time:.2f}с)")
            print(f"   Память: {result.memory_used_mb:.1f} MB")
            if result.wer is not None:
                print(f"   Точность: WER={result.wer:.2%}, CER={result.cer:.2%}")
            print()
        
        # Определяем победителей
        fastest = min(self.results, key=lambda x: x.total_time)
        print(f"🏆 Самый быстрый: {fastest.model_name} ({fastest.total_time:.2f} сек)")
        
        if self.results[0].wer is not None:
            most_accurate = min(self.results, key=lambda x: x.wer)
            print(f"🎯 Самый точный: {most_accurate.model_name} (WER={most_accurate.wer:.2%})")
        
        print(f"\n{'='*80}")


# Пример использования
if __name__ == "__main__":
    from faster_whisper_transcribe import transcribe_with_faster_whisper
    from whisper_transcribe import transcribe_with_whisper
    
    # Путь к тестовому аудио
    audio_file = "test_audio.wav"
    
    # Эталонный текст (если есть) для оценки точности
    reference_text = "Привет мир это тестовое аудио на русском языке"
    
    # Создаем бенчмарк
    benchmark = ASRBenchmark(audio_file, reference_text=reference_text)
    
    # Тестируем Faster-Whisper
    benchmark.benchmark_function(
        transcribe_with_faster_whisper,
        model_name="Faster-Whisper (base, int8, CPU)",
        model_size="base",
        device="cpu",
        compute_type="int8"
    )
    
    # Тестируем оригинальный Whisper
    benchmark.benchmark_function(
        transcribe_with_whisper,
        model_name="Whisper OpenAI (base, CPU)",
        model_size="base",
        device="cpu"
    )
    
    # Выводим сравнение
    benchmark.print_comparison()