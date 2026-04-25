

uv run python -m src.main


nvcc -O3 -arch=sm_86 -shared -Xcompiler -fPIC -o libnms.so nms.cu



pip install pybind11

   python setup.py build_ext --inplace


uv run pip install ultralytics

uv run yolo export model=yolov8n.pt format=onnx

mv yolov8n.onnx models/

uv run python -m src.main
