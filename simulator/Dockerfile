FROM python:3.12-slim

# Prevent Python from buffering stdout so logs stream instantly
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy the simulator script
COPY simulator/simulator.py .

# Expose the default MLLP and Pager ports
EXPOSE 8440 8441

# Run the simulator.
# We point --messages to the /data mount where our sample dataset lives.
CMD ["python", "simulator.py", "--messages", "/data/messages.mllp", "--mllp", "8440", "--pager", "8441"]