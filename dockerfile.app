FROM historical-archive-base:latest

# Copy the application code into the container
COPY app/ /app/app/
COPY pages/ /app/pages/

WORKDIR /app

EXPOSE 5000

CMD ["python", "app/app.py"]