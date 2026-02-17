FROM historical-archive-base:latest

# Copy the application code into the container
COPY src/ /app/src/
COPY data/ /app/data/

WORKDIR /app

EXPOSE 5000

CMD ["python", "src/app/app.py"]