# ============================================================
# STAGE 1: BUILDER (compilação)
# ============================================================
FROM python:3.10-slim as builder

WORKDIR /app
COPY requirements.txt .

# Instalar dependências em ~/.local (reduz tamanho final)
RUN pip install --user --no-cache-dir -r requirements.txt

# ============================================================
# STAGE 2: RUNTIME (imagem final)
# ============================================================
FROM python:3.10-slim

WORKDIR /app

# Copiar apenas site-packages do builder (não o cache)
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copiar apenas o código — modelos e vectorizer são baixados do DagsHub no startup
COPY src/ src/
COPY params.yaml .

# Expor porta da API
EXPOSE 8000

# Comando padrão: rodar API
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]