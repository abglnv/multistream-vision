_PER_SERVER = 2
_BASE_PORT = 8554

STREAMS = [
    f"rtsp://localhost:{_BASE_PORT + i // _PER_SERVER}/stream{i}"
    for i in range(8)
]
