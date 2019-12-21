FROM python:3.7
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install wheel
RUN pip install pandas
RUN pip install -r requirements.txt
RUN pip install opencv-python
COPY finalized_model.sav /finalized_model.sav
COPY ocrscrserv.py /ocrscrserv.py
EXPOSE 80
CMD ["python",  "ocrscrserv.py"]