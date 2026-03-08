# Lap-Trackr Documentation

Capture and grade laparoscopic surgical training sessions using stereo cameras, instrument detection, and 3D kinematic analysis.

## Documentation Index

| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | System architecture, service topology, data flow |
| [Jetson Setup](setup-jetson.md) | First-time setup on a new Jetson AGX Orin |
| [Development Setup](setup-dev.md) | Local development on Windows, Linux, or macOS |
| [Environment Variables](environment-variables.md) | Complete reference for all configuration |
| [API Reference](api-reference.md) | REST endpoints, authentication, request/response formats |
| [Services Deep Dive](services.md) | Internals of each microservice |
| [Database & Migrations](database.md) | PostgreSQL schema ownership, Redis job state, Alembic workflow, backup/restore |
| [Data Model](data-model.md) | Database schema, session directory layout, file formats |
| [Calibration Guide](calibration.md) | Camera calibration workflow and ChArUco board setup |
| [ML Backends](ml-backends.md) | Instrument detection models and the tracking pipeline |
| [Troubleshooting](troubleshooting.md) | Common issues and how to resolve them |

## Quick Links

- **Production (Jetson)**: See [Jetson Setup](setup-jetson.md)
- **Development (any machine)**: See [Development Setup](setup-dev.md)
- **Adding a new ML backend**: See [ML Backends](ml-backends.md#adding-a-new-backend)
- **API integration**: See [API Reference](api-reference.md)
