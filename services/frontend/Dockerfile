# Stage 1: Build
FROM node:20-slim AS builder

WORKDIR /app

# Install dependencies first (layer caching)
COPY package.json package-lock.json* ./
RUN npm install --frozen-lockfile 2>/dev/null || npm install

# Copy source and build
COPY . .
RUN npm run build

# Stage 2: Output
# Lightweight image that copies the built assets into the shared volume at runtime.
FROM alpine:3.19

WORKDIR /app

# Store built assets in a staging directory (not the volume mount point)
COPY --from=builder /app/dist /app/_build

# At runtime, clear the volume and copy fresh assets into it.
# The volume is mounted at /app/dist by docker-compose.
CMD ["sh", "-c", "rm -rf /app/dist/* 2>/dev/null; cp -r /app/_build/* /app/dist/ && echo 'Frontend build deployed to /app/dist'"]
