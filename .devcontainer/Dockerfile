FROM mcr.microsoft.com/vscode/devcontainers/miniconda:0-3

ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/usr/local/lib:/lib:/usr/lib:/opt/conda/lib"

RUN apt-get update && \
    apt-get -y install --no-install-recommends ssh tmux neovim

COPY environment.yml /tmp/conda-tmp/
RUN umask 0002 && \
    /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml && \
    rm -rf /tmp/conda-tmp
