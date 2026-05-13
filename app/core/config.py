from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Device
    DEVICE_ID: str = "RPi-001"
    LOT_ID: str = "lot-001"

    # Database
    DATABASE_URL: str = "sqlite:///data/parking.db"

    # MQTT
    MQTT_BROKER_HOST: str = "15.235.50.88"
    MQTT_BROKER_PORT: int = 41883
    MQTT_USERNAME: str = "admin"
    MQTT_PASSWORD: str = "Broker@123"
    # How often to publish a full slot-state snapshot (retained) for reconciliation.
    # Change events publish immediately to a separate topic; this is the heartbeat.
    MQTT_SNAPSHOT_INTERVAL: int = 300  # seconds (5 minutes)

    # Central API
    CENTRAL_API_URL: str = "http://localhost:8100/api/v1"
    CENTRAL_API_TOKEN: str = ""

    # Detection
    DETECTION_INTERVAL: int = 30
    YOLO_MODEL_PATH: str = "models/yolo26n_ncnn_model"
    YOLO_CONFIDENCE: float = 0.15
    DEPTH_MODEL_PATH: str = "models/onnx/model_quantized.onnx"
    DEPTH_INPUT_SIZE: int = 384

    # Server
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8300


settings = Settings()
