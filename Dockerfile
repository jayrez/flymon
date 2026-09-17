FROM rocker/r-ver:4.3.2

# System dependencies for R spatial/connectomics libraries and headless OpenGL
RUN apt-get update && apt-get install -y \
    git \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    libglu1-mesa-dev freeglut3-dev libx11-dev \
    libgdal-dev libproj-dev libgeos-dev \
    libgit2-dev libgmp-dev libglpk-dev \
    libprotobuf-dev protobuf-compiler libjq-dev \
    xvfb python3-pip python3-venv python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Set Posit Binary repository for Ubuntu 22.04 (Jammy)
ENV R_REPOS="https://packagemanager.posit.co/cran/__linux__/jammy/latest"

# Install pak package manager and benchmark timer
RUN Rscript -e "install.packages(c('pak', 'tictoc'), repos=Sys.getenv('R_REPOS'))"

# Install malecns and all natverse dependencies using pak
RUN Rscript -e "options(repos = c(CRAN = Sys.getenv('R_REPOS'))); pak::pkg_install('natverse/malecns')"

ENV RGL_USE_NULL=TRUE
WORKDIR /workspace