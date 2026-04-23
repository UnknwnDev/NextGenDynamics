FROM nvcr.io/nvidia/isaac-lab:2.3.2

WORKDIR /workspace/SpiderBotAIProject

COPY . /workspace/SpiderBotAIProject

RUN bash -lc " \
    cd /workspace/isaaclab && \
    ./isaaclab.sh -p -m pip install --upgrade pip && \
    ./isaaclab.sh -p -m pip install --upgrade opensimplex && \
    ./isaaclab.sh -p -m pip install 'skrl>=1.4.3' && \
    ./isaaclab.sh -p -m pip install -e /workspace/SpiderBotAIProject/source/SpiderBotAIProject \
"
