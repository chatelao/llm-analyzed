# Galerie der LLM-Architektur (Llama 3.1 & Transformer)

Diese Galerie enthält 10 ausgewählte Grafiken, die die Struktur und Funktionsweise moderner Large Language Models wie Llama 3.1 illustrieren.

---

### 1. Transformer Hochlevel-Struktur
![Transformer Hochlevel-Struktur](https://jalammar.github.io/images/t/the_transformer_3.png)
Eine abstrakte Darstellung des Modells als "Black Box", die Eingabesequenzen verarbeitet und Ausgaben generiert.

### 2. Encoder und Decoder Stacks
![Encoder und Decoder Stacks](https://jalammar.github.io/images/t/The_transformer_encoder_decoder_stack.png)
Llama 3 verwendet eine "Decoder-only" Architektur, aber dieses Diagramm zeigt die ursprüngliche Stapelung von Blöcken.

### 3. Der Transformer Encoder Block
![Transformer Encoder Block](https://jalammar.github.io/images/t/Transformer_encoder.png)
Detaillierte Ansicht eines einzelnen Blocks mit Self-Attention und Feed-Forward Netzwerken.

### 4. Self-Attention Visualisierung
![Self-Attention Visualisierung](https://jalammar.github.io/images/t/transformer_self-attention_visualization.png)
Illustration, wie das Modell beim Kodieren eines Wortes (z.B. "it") Bezüge zu anderen Wörtern im Satz herstellt.

### 5. Self-Attention Vektoren (Q, K, V)
![Self-Attention Vektoren](https://jalammar.github.io/images/t/transformer_self_attention_vectors.png)
Darstellung der Query-, Key- und Value-Vektoren, die die Grundlage der Attention-Berechnung bilden.

### 6. Matrix-Berechnung der Self-Attention
![Matrix-Berechnung der Self-Attention](https://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png)
Die mathematische Zusammenfassung der Attention-Schritte in effizienter Matrixform.

### 7. Multi-Head Attention Zusammenfassung
![Multi-Head Attention Zusammenfassung](https://jalammar.github.io/images/t/transformer_multi-headed_self-attention-recap.png)
Übersicht über die parallele Verarbeitung in mehreren Attention-Heads (Llama 3 nutzt hier GQA).

### 8. Positionale Kodierung (Positional Encoding)
![Positionale Kodierung](https://jalammar.github.io/images/t/transformer_positional_encoding_large_example.png)
Visualisierung der Muster, die dem Modell Informationen über die Reihenfolge der Wörter geben.

### 9. Output-Layer: Logits und Softmax
![Output-Layer](https://jalammar.github.io/images/t/transformer_decoder_output_softmax.png)
Der Prozess der Umwandlung von Vektoren in Wahrscheinlichkeiten für das nächste Wort im Vokabular.

### 10. Maskierte Self-Attention
![Maskierte Self-Attention](https://jalammar.github.io/images/t/self-attention-softmax.png)
Entscheidend für die Inferenz: Das Modell darf bei der Generierung nur auf vergangene, nicht auf zukünftige Tokens schauen.

---
*Quelle der Grafiken: Jay Alammar (jalammar.github.io)*
