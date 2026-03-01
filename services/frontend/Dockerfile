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
# Lightweight image that just copies the built assets and exits
FROM alpine:3.19

WORKDIR /app

# Copy built assets from builder
COPY --from=builder /app/dist /app/dist

# This container's only purpose is to provide the built assets.
# In docker-compose, the dist directory is shared via a volume,
# and the container exits immediately after starting.
CMD ["echo", "Frontend build complete - assets available at /app/dist"]
