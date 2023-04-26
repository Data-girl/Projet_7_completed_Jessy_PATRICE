FROM python:3.9-rc-slim-buster

WORKDIR /app/

COPY . /app/

RUN apt-get update && apt-get install libgomp1

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port=8000"]