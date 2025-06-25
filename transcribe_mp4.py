#!/usr/bin/env python3
"""
Uso en Windows (PowerShell o CMD):
    python transcribe_mp4.py video.mp4 > transcripcion.txt

Requisitos:
    - Python ≥ 3.8
    - ffmpeg en el PATH de Windows
    - pip install openai-whisper ffmpeg-python
"""

import argparse
import logging
import os
import subprocess
import tempfile
import whisper

logging.basicConfig(level=logging.INFO)

def extraer_audio(archivo_video: str, archivo_audio: str) -> None:
    """Extrae el audio a WAV mono 16 kHz con ffmpeg."""
    subprocess.check_call([
        'ffmpeg',
        '-i', archivo_video,
        '-ar', '16000',
        '-ac', '1',
        '-c:a', 'pcm_s16le',
        archivo_audio,
        '-y',
        '-loglevel', 'error'
    ])
    logging.info('Audio extraído a %s', archivo_audio)


def transcribir(archivo_video: str, modelo: str = 'medium', language: str = 'es') -> str:
    """Devuelve la transcripción en español."""
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, 'audio.wav')
        logging.info('Extrayendo audio de %s a %s', archivo_video, wav)

        extraer_audio(archivo_video, wav)

        asr = whisper.load_model(modelo)
        resultado = asr.transcribe(wav, language=language, fp16=False)
        logging.info('Transcripción completada en %s para %s', language, archivo_video)

        return resultado['text']


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Transcribe en español un archivo .mp4 con Whisper.'
    )
    parser.add_argument('video', help='Ruta del archivo .mp4')
    parser.add_argument(
        '-m', '--model',
        default='medium',
        choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2'],
        help='Modelo Whisper.'
    )
    args = parser.parse_args()
    texto = transcribir(args.video, args.model)
    print(texto)


if __name__ == '__main__':
    main()
