FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime AS develop
WORKDIR /workspace
COPY requirements.txt .
RUN set -eux; \
	apt-get update; \
	apt-get install -y curl gcc git libopenslide0; \
	mkdir -p /resources; \
	curl -o /resources/xiyue-wang.pth https://localtoast-dump.s3.eu-central-1.amazonaws.com/xiyue-wang.pth; \
	pip install -r requirements.txt

FROM develop AS deploy
RUN cp /resources/xiyue-wang.pth /workspace
COPY . /workspace
ENTRYPOINT [ \
	"python3","create_heatmaps.py", \
	"--model-path", "/model/export.pkl", \
	"--output-path", "/output", \
	"--cache-dir", "/cache" ]
