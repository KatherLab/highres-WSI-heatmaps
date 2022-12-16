FROM nvcr.io/nvidia/pytorch:22.11-py3 AS develop
RUN set -exu; \
	apt-get update; \
	DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libopenslide0; \
	rm -rf /var/lib/apt/lists/; \
	mkdir -p /resources; \
	curl -o /resources/xiyue-wang.pth https://localtoast-dump.s3.eu-central-1.amazonaws.com/xiyue-wang.pth
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
