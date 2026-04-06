# Python 3.10 slim 버전을 사용하여 기본 이미지 크기를 최소화합니다
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# OpenCV 및 관련 시스템 의존성 설치 (-y 및 --no-install-recommends로 최적화, 캐시 지우기)
# 1. 캐시를 삭제하고 업데이트를 시도합니다.
RUN apt-get clean && \
    apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 요구사항 파일 복사 (의존성이 변경되지 않은 경우 Docker 캐시를 활용하기 위해 먼저 복사)
COPY requirements.txt .

# pip 업그레이드 및 파이썬 패키지 설치 (--no-cache-dir 옵션으로 불필요한 캐시 제거)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# [최적화] 첫 번째 요청 시 지연을 방지하기 위해 컨테이너 빌드 프로세스 중에
# DeepFace 모델 가중치를 미리 다운로드합니다.
# 더미 이미지를 넘겨 모델 다운로드를 트리거합니다.
RUN python -c "from deepface import DeepFace; import numpy as np; \
    try: DeepFace.analyze(np.zeros((224, 224, 3), dtype=np.uint8), actions=['age'], enforce_detection=False) \
    except Exception: pass"

# 애플리케이션 코드 복사
COPY main.py .

# FastAPI 서버가 사용할 포트 개방
EXPOSE 8000

# uvicorn으로 FastAPI 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
