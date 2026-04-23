# Use the minimal Isaac Lab image as base
FROM nvcr.io/nvidia/isaac-lab:2.3.0

# Project workspace for the external project
WORKDIR /workspace/project

# Install Python dependencies
COPY requirements.txt ./requirements.txt
RUN /isaac-sim/python.sh -m pip install --no-cache-dir -r requirements.txt

# Copy the entire external project into the image
COPY . .

# Install the external project (ChargeProject) in editable mode
# This assumes your package lives at: source/ChargeProject
RUN /isaac-sim/python.sh -m pip install -e source/ChargeProject

# Switch to the Isaac Lab root directory (isaaclab.sh lives here)
WORKDIR /workspace/IsaacLab

# Development entrypoint:
#   - Start an interactive shell
#   - Default command keeps the container alive (good for Runpod / VS Code attach)
ENTRYPOINT ["bash", "-lc"]
CMD ["sleep infinity"]
