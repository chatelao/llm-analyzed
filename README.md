# Analyse der LLM-Verarbeitung (Llama 3.1 405B)

Diese Analyse berechnet die Anzahl der Multiplikationen (MACs), die für die Verarbeitung einer Abfrage mit 1 Million Eingabe-Token auf dem derzeit größten Modell in der Ollama-Bibliothek erforderlich sind.

## 1. Modell-Parameter (Llama 3.1 405B)

| Parameter | Symbol | Wert | Quelle |
| :--- | :--- | :--- | :--- |
| Layer | $L$ | 126 | Meta Llama 3.1 Paper [[1]] |
| Hidden Dimension | $d_{model}$ | 16.384 | Meta Llama 3.1 Paper [[1]] |
| FFN Dimension | $d_{ff}$ | 53.248 | Meta Llama 3.1 Paper [[1]] |
| Attention Query Heads | $h$ | 128 | Meta Llama 3.1 Paper [[1]] |
| Attention KV Heads | $h_{kv}$ | 8 | Meta Llama 3.1 Paper [[1]] |
| Vokabulargröße | $V$ | 128.256 | Meta Llama 3.1 Paper [[1]] |
| Kontext-Länge | $N$ | 1.000.000 | Benutzeranfrage |

---

## 2. Mathematischer Ablauf und Berechnungsformeln

Wir berechnen die Multiply-Accumulate Operations (MACs) für die **Prefill-Phase**.

### Schritt A: Lineare Projektionen (Attention)
In jedem Layer werden die Token in Q, K, V und O projiziert.
```math
MACs_{attn\_proj} = N \cdot (2 \cdot d_{model}^2 + 2 \cdot (d_{model} \cdot d_{head} \cdot h_{kv}))
```
*Hinweis:
```math
d_{head} = d_{model} / h = 128$.*
```

### Schritt B: Attention-Mechanik (Quadratisch)
Berechnung der Scores ($Q K^T$) und des Kontextvektors ($S V$).
```math
MACs_{attn\_mech} = 2 \cdot (N^2 \cdot d_{model})
```

### Schritt C: Feed-Forward Network (MLP)
Llama nutzt SwiGLU mit drei Matrizen ($W_{gate}, W_{up}, W_{down}$).
```math
MACs_{mlp} = N \cdot (3 \cdot d_{model} \cdot d_{ff})
```

### Schritt D: Unembedding (Output Layer)
Projektion des finalen Hidden State auf das Vokabular.
```math
MACs_{output} = N \cdot d_{model} \cdot V
```

---

## 3. Berechnung für 1 Million Token ($10^6$)

| Komponente | Formel | Multiplikationen (MACs) |
| :--- | :--- | :--- |
| **Linear (Proj + MLP)** | $L \cdot (MACs_{attn\_proj} + MACs_{mlp})$ | $4,02 \cdot 10^{17}$ |
| **Attention (quadr.)** | $L \cdot MACs_{attn\_mech}$ | $4,13 \cdot 10^{18}$ |
| **Output Head** | $MACs_{output}$ | $2,10 \cdot 10^{15}$ |
| **Gesamt** | | **$4,53 \cdot 10^{18}$** |

---

## 4. Quellenangaben

*   [[1]] **Dubey et al. (2024):** *"The Llama 3 Herd of Models"*. Meta AI Technical Report. (Spezifikationen für 405B Architektur).
*   [[2]] **Kaplan et al. (2020):** *"Scaling Laws for Neural Language Models"*. (Standardformeln für Transformer-Komplexität).
*   [[3]] **Vaswani et al. (2017):** *"Attention Is All You Need"*. (Quadratic Scaling of Dot-Product Attention).
*   [[4]] **Ollama Model Library:** `llama3.1:405b` Manifest (Verifikation der Context Window und Layer-Struktur).
