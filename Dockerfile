FROM python:3.7
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install wheel
RUN pip install pandas
RUN pip install -r requirements.txt
RUN pip install opencv-python
COPY name.jpg /name.jpg
COPY ocrscrserv.py /ocrscrserv.py
EXPOSE 8088
CMD ["python",  "ocrscrserv.py"]
