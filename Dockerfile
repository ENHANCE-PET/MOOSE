# Base image
FROM python:3.10-slim

# Install MOOSE from PyPI
RUN pip install moosez

# Set working directory
WORKDIR /app

# Entry point for the MOOSE CLI
ENTRYPOINT ["moosez"]

# Default command
CMD ["-h"]
