from __future__ import annotations

import json
import os

import boto3

_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID",
    "anthropic.claude-sonnet-4-5-20250929-v1:0",
)


def generate_ai_report(summary: dict) -> str:
    """Call Bedrock Runtime with the vocal analysis summary and return the AI report."""
    region = os.environ.get("AWS_REGION", "us-east-2")
    client = boto3.client("bedrock-runtime", region_name=region)

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": _build_prompt(summary)}],
    })

    response = client.invoke_model(
        modelId=_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    data = json.loads(response["body"].read())
    return data["content"][0]["text"]


def _build_prompt(summary: dict) -> str:
    metrics_json = json.dumps(summary, ensure_ascii=False, indent=2)
    return f"""Eres un profesor experto de canto clásico y popular. Analiza las siguientes métricas de rendimiento vocal de un estudiante y genera un informe pedagógico detallado en español.

## Métricas del análisis vocal:
```json
{metrics_json}
```

## Referencia de métricas y valores ideales:
- **mean_note_accuracy_pct**: precisión de afinación promedio (excelente > 85%, bueno 70-85%, regular 55-70%, necesita mejora < 55%)
- **wobble_score**: inestabilidad involuntaria 0-100 (excelente < 20, bueno 20-35, regular 35-50, problemático > 50)
- **vibrato_rate_hz**: velocidad del vibrato en Hz (natural 5.5-7.0 Hz; lento < 5.5, rápido > 7.0)
- **vibrato_extent_hz**: amplitud del vibrato en Hz (ideal 20-50 Hz; estrecho < 20, exagerado > 50)
- **vibrato_class**: clasificación del vibrato (natural/controlado, lento, rápido, irregular, ausente)
- **rms_cv**: variación de dinámica (bajo < 0.3: poca dinámica, medio 0.3-0.6: normal, alto > 0.6: muy expresivo)
- **breath_control**: clasificación del control del aliento basada en estabilidad de volumen
- **brightness**: calidad tímbrica espectral (oscuro, equilibrado, brillante)
- **tone_quality**: calidad del timbre (tono limpio vs presencia de aire/tensión)
- **range_semitones**: rango vocal en semitonos (estrecho < 12, normal 12-24, amplio > 24)
- **voice_percentage**: porcentaje del audio con voz activa detectada (ideal > 65%)
- **vocal_grade**: calificación final calculada 1-10

## Formato del informe requerido:

### Resumen General
(2-3 oraciones de evaluación global hablándole directamente al estudiante)

### Análisis por Métrica
(Para cada métrica importante: qué mide, valor obtenido, nivel alcanzado y qué significa musicalmente)

### Puntos Fuertes
- (2-3 aspectos donde el estudiante destaca, con valores concretos)

### Áreas de Mejora
- (2-3 aspectos prioritarios, con explicación del impacto musical y por qué importan)

### Calificación: X/10
(Justificación pedagógica de la nota basada en las métricas obtenidas)

### Ejercicios Recomendados
1. **[Nombre]**: descripción breve y objetivo específico
(3-5 ejercicios concretos dirigidos a las debilidades identificadas)

---
Usa un tono profesional y alentador. Habla directamente al estudiante ("Tu vibrato...", "Tu afinación..."). Cita los valores numéricos obtenidos para que el estudiante entienda su progreso."""
