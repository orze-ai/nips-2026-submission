FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    conda clean -afy

# Accept conda TOS
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Create conda environment and install dependencies
SHELL ["/bin/bash", "-c"]
RUN conda create -n vlm_env python=3.11 -y
RUN source activate vlm_env && \
    pip install torch==2.9.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
RUN source activate vlm_env && \
    pip install -e .

# Set conda environment activation in bashrc
RUN echo "source activate vlm_env" >> ~/.bashrc

# Default command
SHELL ["conda", "run", "-n", "vlm_env", "/bin/bash", "-c"]

# Expose port for wandb (optional)
EXPOSE 8080

# Default entrypoint
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "vlm_env"]
CMD ["/bin/bash"]
