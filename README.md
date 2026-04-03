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

### 1.1 Erläuterung der Variablen

*   **$L$ (Layer):** Die Anzahl der Transformer-Blöcke (Layer). Jeder Layer führt eine vollständige Sequenz aus Attention- und Feed-Forward-Berechnungen durch. Ein tieferes Modell kann komplexere Zusammenhänge lernen.
*   **$d_{model}$ (Hidden Dimension):** Die Breite des Modells bzw. die Größe der Vektordarstellung (Embedding) jedes Tokens. Sie bestimmt die Kapazität des Modells, Informationen pro Token zu kodieren.
*   **$d_{ff}$ (FFN Dimension):** Die Dimension der Zwischenschicht im Feed-Forward-Netzwerk (MLP). Llama 3 nutzt die SwiGLU-Architektur, bei der die Dimension $d_{ff}$ die Größe der versteckten Schichten innerhalb dieses Moduls definiert.
*   **$h$ (Attention Query Heads):** Die Anzahl der Köpfe für die "Queries" in der Multi-Head Attention. Dies ermöglicht es dem Modell, parallel verschiedene Arten von Abhängigkeiten im Text zu verarbeiten.
*   **$h_{kv}$ (Attention KV Heads):** Die Anzahl der Köpfe für "Keys" und "Values". Llama 3 verwendet *Grouped-Query Attention* (GQA), wobei sich mehrere Query-Heads einen KV-Head teilen, um die Effizienz (insbesondere den Speicherbedarf) zu erhöhen.
*   **$V$ (Vokabulargröße):** Die Gesamtanzahl der eindeutigen Tokens, die das Modell in seinem Wörterbuch führt. Dies beeinflusst die Größe der finalen Ausgabeschicht (Unembedding).
*   **$N$ (Kontext-Länge):** Die Anzahl der Token in der aktuellen Eingabesequenz. Da die Attention-Mechanik quadratisch mit $N$ skaliert, ist dies der kritischste Faktor für die Rechenlast bei langen Texten.
*   **$d_{head}$ (Head Dimension):** Die Dimension eines einzelnen Attention-Kopfes ($d_{model} / h$). Sie bestimmt die Größe der Vektoren, die im Skalarprodukt der Attention-Berechnung miteinander multipliziert werden.

---

## 2. Mathematischer Ablauf und Berechnungsformeln

Wir berechnen die Multiply-Accumulate Operations (MACs) für die **Prefill-Phase**.

### Schritt A: Lineare Projektionen (Attention)
In jedem Layer werden die Token in Q, K, V und O projiziert.
```math
MACs_{attn\_proj} = N \cdot (2 \cdot d_{model}^2 + 2 \cdot (d_{model} \cdot d_{head} \cdot h_{kv}))
```
```math
Elements_{fetch} = (2 \cdot d_{model}^2 + 2 \cdot (d_{model} \cdot d_{head} \cdot h_{kv})) + N \cdot d_{model}
```
```math
Elements_{store} = N \cdot d_{model}
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
```math
Elements_{fetch} = N \cdot d_{model} + h \cdot N^2
```
```math
Elements_{store} = h \cdot N^2 + N \cdot d_{model}
```

### Schritt C: Feed-Forward Network (MLP)
Llama nutzt SwiGLU mit drei Matrizen ($W_{gate}, W_{up}, W_{down}$).
```math
MACs_{mlp} = N \cdot (3 \cdot d_{model} \cdot d_{ff})
```
```math
Elements_{fetch} = (3 \cdot d_{model} \cdot d_{ff}) + N \cdot d_{model}
```
```math
Elements_{store} = N \cdot d_{model}
```

### Schritt D: Unembedding (Output Layer)
Projektion des finalen Hidden State auf das Vokabular.
```math
MACs_{output} = N \cdot d_{model} \cdot V
```
```math
Elements_{fetch} = (d_{model} \cdot V) + N \cdot d_{model}
```
```math
Elements_{store} = N \cdot V
```

---

## 3. Berechnung für 1 Million Token ($10^6$)

| Komponente | Formel | Multiplikationen (MACs) | Read/Fetch (Elemente) | Write/Store (Elemente) |
| :--- | :--- | :--- | :--- | :--- |
| **Linear (Proj + MLP)** | $L \cdot (MACs_{attn\_proj} + MACs_{mlp})$ | $4,02 \cdot 10^{17}$ | $4,53 \cdot 10^{12}$ | $4,13 \cdot 10^{12}$ |
| **Attention (quadr.)** | $L \cdot MACs_{attn\_mech}$ | $4,13 \cdot 10^{18}$ | $1,61 \cdot 10^{16}$ | $1,61 \cdot 10^{16}$ |
| **Output Head** | $MACs_{output}$ | $2,10 \cdot 10^{15}$ | $1,85 \cdot 10^{10}$ | $1,28 \cdot 10^{11}$ |
| **Gesamt** | | **$4,53 \cdot 10^{18}$** | **$1,61 \cdot 10^{16}$** | **$1,61 \cdot 10^{16}$** |

---

## 4. Quellenangaben

*   [[1]] **Dubey et al. (2024):** *"The Llama 3 Herd of Models"*. Meta AI Technical Report. (Spezifikationen für 405B Architektur).
*   [[2]] **Kaplan et al. (2020):** *"Scaling Laws for Neural Language Models"*. (Standardformeln für Transformer-Komplexität).
*   [[3]] **Vaswani et al. (2017):** *"Attention Is All You Need"*. (Quadratic Scaling of Dot-Product Attention).
*   [[4]] **Ollama Model Library:** `llama3.1:405b` Manifest (Verifikation der Context Window und Layer-Struktur).
