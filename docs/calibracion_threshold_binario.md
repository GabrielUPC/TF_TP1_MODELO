# Calibración y threshold del modelo binario

Esta fase analiza exclusivamente las probabilidades out-of-fold producidas por
el backtesting temporal de 2018-2024. No vuelve a ajustar los tres modelos, no
abre 2025/2026 y no modifica ningún artefacto productivo.

La calibración evaluada es cruda: Brier score, diez intervalos uniformes,
probabilidad media, prevalencia observada, error absoluto ponderado (ECE) y
sesgo firmado. No se ajusta sigmoid/Platt ni isotonic. Probar esos métodos exige
otra validación temporal específica.

La grilla de decisión contiene 76 valores entre 0.05 y 0.80, con paso 0.01. En
cada valor se calculan TP, TN, FP, FN, recall, especificidad, PPV, NPV, F1,
balanced accuracy, accuracy, tasa de alertas y tasa de falsos negativos.

Los escenarios son: máximo F1, máxima balanced accuracy, mayor threshold capaz
de conservar recall mínimo 0.80, 0.85 o 0.90 y referencia 0.50. Elegir el mayor
threshold factible para un recall mínimo busca reducir alertas innecesarias sin
incumplir esa sensibilidad en los datos de selección.

La validación principal de la regla de decisión es walk-forward. Para evaluar
2019 se selecciona con 2018; para evaluar 2020 se selecciona con 2018-2019, y
así sucesivamente hasta 2024. Por tanto, el año evaluado nunca participa en la
elección de su threshold. El resumen agregado 2018-2024 solo propone candidatos
para una evaluación futura aún sellada; no constituye un threshold operacional
definitivo.
