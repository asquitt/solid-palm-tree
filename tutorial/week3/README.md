# Week 3: Model Serving and APIs

Deploy your fine-tuned models as production REST APIs.

## Goals
- ✅ Deploy models with Ray Serve
- ✅ Build FastAPI endpoints
- ✅ Implement request batching
- ✅ Handle concurrent requests
- ✅ Add API versioning

## Files
- `starter/model_server.py` - Model serving implementation
- `starter/api_endpoints.py` - FastAPI routes
- `solution/` - Reference implementations

## Key Concepts
- Ray Serve deployment
- Request batching for efficiency
- Async request handling
- API design best practices

## Getting Started
```bash
cd starter
# Implement model_server.py
python model_server.py  # Start server
curl http://localhost:8000/predict  # Test
```

## Time Estimate: 8-10 hours
