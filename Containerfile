FROM pytorch/pytorch AS develop
WORKDIR /resources
RUN apt-get update \
	&& apt-get install -y gcc git python3-dev libopenslide0 wget \
	&& wget https://localtoast-dump.s3.eu-central-1.amazonaws.com/xiyue-wang.pth
COPY requirements.txt .
RUN pip install -r requirements.txt
WORKDIR /workspace

FROM develop AS deploy
RUN cp /resources/xiyue-wang.pth /workspace
COPY . /workspace
ENTRYPOINT [ \
	"python3","create_heatmaps.py", \
	"--model-path", "/model/export.pkl", \
	"--output-path", "/output", \
	"--cache-dir", "/cache" ]
