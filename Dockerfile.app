FROM python:3.9

WORKDIR /display_ui

COPY . .

RUN pip install -r requirements.txt

CMD [ "python", "app.py" ]
