from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Device
    DEVICE_ID: str = "RPi-001"

    # Database
    DATABASE_URL: str = "sqlite:///data/parking.db"

    # MQTT
    MQTT_BROKER_HOST: str = "15.235.50.88"
    MQTT_BROKER_PORT: int = 41883
    MQTT_USERNAME: str = "admin"
    MQTT_PASSWORD: str = "Broker@123"
    MQTT_SNAPSHOT_INTERVAL: int = 300  # seconds (5 minutes)

    # Detection
    DETECTION_INTERVAL: int = 30
    DETECTION_DEBOUNCE_ENABLED: bool = True
    DETECTION_DEBOUNCE_COUNT: int = 3  # consecutive same-state detections before reporting change
    YOLO_MODEL_PATH: str = "models/yolo26n_ncnn_model"
    YOLO_CONFIDENCE: float = 0.15
    DEPTH_MODEL_PATH: str = "models/onnx/model_quantized.onnx"
    DEPTH_INPUT_SIZE: int = 384

    # MinIO
    MINIO_ENDPOINT: str = "api-minio.projectanddemoserver.com"
    MINIO_ACCESS_KEY: str = ""
    MINIO_SECRET_KEY: str = ""
    MINIO_SECURE: bool = True
    MINIO_BUCKET: str = "ai-parking"

    # Server
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8300


settings = Settings()
