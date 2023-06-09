FROM python:3.8

ADD new_theory.py .
ADD utils.py .

RUN pip install pandas numpy boto3 ta tqdm

RUN apt-get update && \
    apt-get install -y awscli

# Set environment variables for AWS access keys
ENV AWS_ACCESS_KEY_ID='AKIA5EFPITXQ4GVIY7HN'
ENV AWS_SECRET_ACCESS_KEY='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'
# Download the file containing the class object from S3

CMD ["python","./new_theory.py"]

