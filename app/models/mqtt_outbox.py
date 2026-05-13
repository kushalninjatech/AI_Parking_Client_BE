from sqlalchemy import Column, Integer, Text, DateTime
from sqlalchemy.sql import func

from app.db.base import Base


class MqttOutbox(Base):
    __tablename__ = "mqtt_outbox"

    id = Column(Integer, primary_key=True, autoincrement=True)
    topic = Column(Text, nullable=False)
    payload = Column(Text, nullable=False)  # JSON string
    created_at = Column(DateTime, server_default=func.now())
    attempts = Column(Integer, default=0)
