FROM public.ecr.aws/sam/build-python3.9:latest

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

ENTRYPOINT python app.py