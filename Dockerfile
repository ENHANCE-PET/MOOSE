# Base image
FROM python:3.13-slim

ARG MOOSEZ_VERSION

# Install the exact MOOSE release from PyPI.
RUN python -m pip install --no-cache-dir "moosez==${MOOSEZ_VERSION}"

# Set working directory
WORKDIR /app

# Entry point for the MOOSE CLI
ENTRYPOINT ["moosez"]

# Default command
CMD ["-h"]
